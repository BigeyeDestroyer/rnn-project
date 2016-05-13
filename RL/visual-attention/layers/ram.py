import theano.tensor as T
import sys
sys.path.append('..')
from common.utils import *
import theano.tensor.signal.pool as pool
import theano.tensor.nnet.abstract_conv as upsample
from theano.tensor.shared_randomstreams import RandomStreams
import math
import numpy


class RAM(object):
    def __init__(self, img_batch, normLoc, y, batch_size=16, mnist_size=28,
                 channels=1, depth=3, minRadius=4, sensorBandwidth=8,
                 hg_size=128, hl_size=128, g_size=256,
                 cell_size=256, cell_out_size=256, n_classes=10,
                 loc_sd=0.01, glimpses=6, numpy_rng=numpy.random.RandomState(1234)):
        """

        :param img_batch: always the tensor variable like T.matrix('img')
        :param normLoc: always the tensor variable like T.matrix('loc')
        :param y: always the tensor variable like T.ivector('label')
        """
        self.img_batch = img_batch
        self.normLoc = normLoc
        self.y = y

        self.batch_size = batch_size
        self.mnist_size = mnist_size
        self.channels = channels
        self.depth = depth
        self.minRadius = minRadius
        self.sensorBandwidth = sensorBandwidth

        self.hg_size = hg_size
        self.hl_size = hl_size
        self.g_size = g_size

        self.cell_size = cell_size
        self.cell_out_size = cell_out_size
        self.n_classes = n_classes

        self.loc_sd = loc_sd
        self.glimpses = glimpses
        self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.totalSensorBandwidth = depth * channels * ((2 * sensorBandwidth) ** 2)

        def _initial_W_b(rng, n_in, n_out):
            """ This inline function initializes
                the params for a single layer MLP

            :param n_in: input dimension
            :param n_out: output dimension
            """
            W = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
                dtype=theano.config.floatX)
            b = numpy.zeros((n_out,), dtype=theano.config.floatX)
            return W, b

        # Initialize the shared variables
        # 1-st set of params: W_hl, b_hl, params for the hl_net
        W_values, b_values = _initial_W_b(rng=numpy_rng,
                                          n_in=2, n_out=self.hl_size)
        self.W_hl = theano.shared(value=W_values, name='hl_net#W', borrow=True)
        self.b_hl = theano.shared(value=b_values, name='hl_net#b', borrow=True)

        # 2-nd set of params: W_hg, b_hg, params for the hg_net
        W_values, b_values = _initial_W_b(rng=numpy_rng,
                                          n_in=self.totalSensorBandwidth,
                                          n_out=self.hg_size)
        self.W_hg = theano.shared(value=W_values, name='hg_net#W', borrow=True)
        self.b_hg = theano.shared(value=b_values, name='hg_net#b', borrow=True)

        # 3-rd set of params: W_g, b_g, params for the h_g net
        W_values, b_values = _initial_W_b(rng=numpy_rng,
                                          n_in=self.hl_size + self.hg_size,
                                          n_out=self.g_size)
        self.W_g = theano.shared(value=W_values, name='g_net#W', borrow=True)
        self.b_g = theano.shared(value=b_values, name='g_net#b', borrow=True)

        # 4-th set of params: W_lstm, U_lstm, params for the LSTMCore
        self.W_lstm = init_weights(shape=(self.g_size, 4 * self.cell_size),
                                   name='LSTMCore#W')
        self.U_lstm = theano.shared(
            value=numpy.concatenate((ortho_weight(ndim=self.cell_size),
                                     ortho_weight(ndim=self.cell_size),
                                     ortho_weight(ndim=self.cell_size),
                                     ortho_weight(ndim=self.cell_size)),
                                    axis=1), name='LSTMCore#U')
        self.b_lstm = init_bias(size=4 * self.cell_size, name='LSTMCore#b')

        # 5-th set of params: W_hl_out, b_hl_out, params for the hl_out net
        W_values, b_values = _initial_W_b(rng=numpy_rng,
                                          n_in=self.cell_size,
                                          n_out=2)
        self.W_hl_out = theano.shared(value=W_values, name='hl_out#W', borrow=True)
        self.b_hl_out = theano.shared(value=b_values, name='hl_out#b', borrow=True)

        # 6-th set of params: W_ha_out, b_ha_out, params for the ha_out net
        W_values, b_values = _initial_W_b(rng=numpy_rng,
                                          n_in=self.cell_size,
                                          n_out=self.n_classes)
        self.W_ha_out = theano.shared(value=W_values, name='ha_out#W', borrow=True)
        self.b_ha_out = theano.shared(value=b_values, name='ha_out#b', borrow=True)

        # Sensor of RAM
        def _glimpseSensor(img_batch, normLoc):
            """ This function calculate the glimpse sensors
                for a batch of images

            :type img_batch: tensor variable with size (batch_size, 784)
            :param img_batch: batch of images

            :type normLoc: tensor variable with size (batch_size, 2)
            :param normLoc: locations of the batch of images

            :return:
            :type zooms: tensor variable with size
                         (batch, depth, channel, height, width)
            :param zooms: zooms of the batch of images
            """
            # from [-1.0, 1.0] -> [0, 28]
            loc = ((normLoc + 1) / 2) * self.mnist_size
            loc = T.cast(loc, 'int32')

            # img with size (batch, channels, height, width)
            img = T.reshape(img_batch, (self.batch_size, self.channels,
                                        self.mnist_size, self.mnist_size))
            # with size (batch, h, w, 1) after reshape

            zooms = []  # zooms of all the images in batch

            maxRadius = self.minRadius * (2 ** (self.depth - 1))  # radius of the largest zoom
            offset = maxRadius

            # zero-padding the batch to (batch, h + 2R, w + 2R, channels)
            img = T.concatenate((T.zeros((self.batch_size, self.channels, maxRadius, self.mnist_size)), img), axis=2)
            img = T.concatenate((img, T.zeros((self.batch_size, self.channels, maxRadius, self.mnist_size))), axis=2)
            img = T.concatenate((T.zeros((self.batch_size, self.channels,
                                          self.mnist_size + 2 * maxRadius, maxRadius)), img), axis=3)
            img = T.concatenate((img, T.zeros((self.batch_size, self.channels,
                                               self.mnist_size + 2 * maxRadius, maxRadius))), axis=3)
            img = T.cast(img, dtype=theano.config.floatX)

            for k in xrange(self.batch_size):
                imgZooms = []  # zoom for a single image

                # one_img with size (channels, 2R + size, 2R + size), channels=1 here
                one_img = img[k, :, :, :]

                for i in xrange(self.depth):
                    # r = minR, 2 * minR, ..., (2^(depth - 1)) * minR
                    r = self.minRadius * (2 ** i)

                    d_raw = 2 * r  # patch size to be cropped

                    loc_k = loc[k, :]  # location of the k-th glimpse, (2, )
                    adjusted_loc = T.cast(offset + loc_k - r, 'int32')  # upper-left corner of the patch

                    # Get a zoom patch with size (d_raw, d_raw) from one_image
                    zoom = one_img[:, adjusted_loc[0]: (adjusted_loc[0] + d_raw),
                           adjusted_loc[1]: (adjusted_loc[1] + d_raw)]

                    if r < self.sensorBandwidth:  # bilinear-interpolation
                        #  here, zoom is a 2D patch with size (2r, 2r)
                        # zoom = T.swapaxes(zoom, 1, 2)
                        # zoom = T.swapaxes(zoom, 0, 1)  # here, zoom with size (channel, height, width)
                        zoom_reshape = T.reshape(zoom, (1, zoom.shape[0], zoom.shape[1], zoom.shape[2]))
                        zoom_bandwidth = upsample.bilinear_upsampling(zoom_reshape,
                                                                      ratio=(self.sensorBandwidth / r),
                                                                      batch_size=1, num_input_channels=self.channels)
                        # bandwith is with size (channel, height, width)
                        zoom_bandwidth = T.reshape(zoom_bandwidth, (zoom_bandwidth.shape[1],
                                                                    zoom_bandwidth.shape[2],
                                                                    zoom_bandwidth.shape[3]))
                    elif r > self.sensorBandwidth:
                        # pooling operation will be down over the last 2 dimensions
                        zoom_bandwidth = pool.pool_2d(input=zoom,
                                                      ds=(r / self.sensorBandwidth,
                                                          r / self.sensorBandwidth),
                                                      mode='average_inc_pad',
                                                      ignore_border=True)
                    else:
                        zoom_bandwidth = zoom

                    imgZooms.append(zoom_bandwidth)

                zooms.append(T.stack(imgZooms))

            # returned zooms is with size (batch, depth, channel, height, width)
            return T.stack(zooms)

        def _slice(x, n, dim):
            if x.ndim == 3:
                return x[:, :, n * dim: (n + 1) * dim]
            return x[:, n * dim: (n + 1) * dim]

        def _step(l_tm1, sampled_l_tm1, c_tm1, h_tm1, x_t):
            """ This function carries on one step operation of RAM

            :type l_tm1: tensor variable
            :param l_tm1: mean location at last time

            :type sampled_l_tm1: tensor variable
            :param sampled_l_tm1: sampled location at last time

            :type c_tm1: tensor variable
            :param c_tm1: c in LSTM at last time

            :type h_tm1: tensor variable
            :param h_tm1: h in LSTM at last time

            :type x_t: tensor variable
            :param x_t: image batch input at current time
            """

            # 1-st part, hl_net: with output size (batch, hl_size)
            hl_output = T.nnet.relu(T.dot(sampled_l_tm1, self.W_hl) + self.b_hl)

            # 2-nd part, hg_net: with output size (batch, hg_size)
            # self.totalSensorBandwidth
            glimpse_input = _glimpseSensor(x_t, sampled_l_tm1)
            hg_input = T.reshape(glimpse_input, (glimpse_input.shape[0],
                                                 self.totalSensorBandwidth))
            hg_output = T.nnet.relu(T.dot(hg_input, self.W_hg) + self.b_hg)

            # 3-rd part, g_net: with output size (batch, g_size)
            g_output = T.nnet.relu(T.dot(T.concatenate((hl_output, hg_output),
                                                       axis=1), self.W_g) + self.b_g)

            # 4-th part, LSTMCore: with output size (batch, cell_size)
            preact = T.dot(g_output, self.W_lstm) + T.dot(h_tm1, self.U_lstm) + self.b_lstm

            i = T.nnet.sigmoid(_slice(preact, 0, self.cell_size))
            f = T.nnet.sigmoid(_slice(preact, 1, self.cell_size))
            o = T.nnet.sigmoid(_slice(preact, 2, self.cell_size))
            c_tilde = T.tanh(_slice(preact, 3, self.cell_size))

            c_t = f * c_tm1 + i * c_tilde
            h_t = o * T.tanh(c_t)

            # 5-th part, hl_out net: with output size (batch, 2)
            l_t = T.tanh(T.dot(h_t, self.W_hl_out) + self.b_hl_out)

            # sampled_l_t is used for next input
            sampled_l_t = l_t + self.theano_rng.normal(size=(self.batch_size, 2),
                                                       avg=0, std=self.loc_sd)
            return l_t, sampled_l_t, c_t, h_t

        def _gaussian_pdf(mean, sample):
            """
            :type mean: tensor variable with size (batch, 2)
            :param mean: mean locations of the images in batch

            :type sample: tensor variable with size (batch, 2)
            :param sample: sampled locations of the images in batch
            """
            Z = 1.0 / (self.loc_sd * T.sqrt(2.0 * math.pi))
            exp_term = -T.square(sample - mean) / (2.0 * T.square(self.loc_sd))
            return Z * T.exp(exp_term)

        [l, l_sampled, c, h], _ = theano.scan(fn=_step,
                                              outputs_info=[self.normLoc,
                                                            self.normLoc,
                                                            T.alloc(floatX(0.), self.img_batch.shape[0], self.cell_size),
                                                            T.alloc(floatX(0.), self.img_batch.shape[0], self.cell_size)],
                                              non_sequences=self.img_batch,
                                              n_steps=self.glimpses)

        # Calculate the cost-related items
        outputs = T.reshape(h[-1], (batch_size, cell_out_size))
        self.p_y_given_x = T.nnet.softmax(T.dot(outputs, self.W_ha_out) + self.b_ha_out)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        R = T.cast(T.eq(self.y_pred, self.y), theano.config.floatX)  # with size (batch, )
        self.reward = T.mean(R)  # average reward of the batch

        p_loc = _gaussian_pdf(mean=l, sample=l_sampled)
        p_loc = T.reshape(p_loc, (self.batch_size, 2 * self.glimpses))
        R = T.reshape(R, (self.batch_size, 1))  # reshape for furthur calculation

        J = T.concatenate((T.reshape(T.log(self.p_y_given_x)[T.arange(self.y.shape[0]), self.y], (self.y.shape[0], 1)),
                           T.reshape(T.mean(T.log(p_loc) * R, axis=1), (self.batch_size, 1))), axis=1)
        self.cost = -T.mean(T.mean(J, axis=1))












