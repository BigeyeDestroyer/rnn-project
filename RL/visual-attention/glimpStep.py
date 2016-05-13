import theano.tensor as T
import numpy
import theano
import sys
sys.path.append('..')
from common.utils import *
from common import mnist_loader
import h5py
from layers.HiddenLayer import HiddenLayer
import theano.tensor.signal.pool as pool
import theano.tensor.nnet.abstract_conv as upsample
from theano.tensor.shared_randomstreams import RandomStreams

# images and labels are the inputs
dataset = mnist_loader.read_data_sets("data")
images, labels = dataset.train.next_batch(batch_size=16)
locs = numpy.random.uniform(low=-1, high=1, size=(images.shape[0], 2))
print images.shape
print labels.shape

# images and locs as inputs


# img : (batch_size, 784)
# normLoc : (batch_size, 2)
img_batch = T.matrix('img')
normLoc = T.matrix('loc')

batch_size = 16
mnist_size = 28
channels = 1
depth = 3
minRadius = 4
sensorBandwidth = 8

hg_size = 128
hl_size = 128
g_size = 256

cell_size = 256  # dimension of the LSTM's hidden state
cell_out_size = cell_size

loc_sd = 0.1


totalSensorBandwidth = depth * channels * ((2 * sensorBandwidth) ** 2)
""" Sensor
"""
def glimpseSensor(img_batch, normLoc, batch_size=16, mnist_size=28,
                  channels=1, depth=3, minRadius=4, sensorBandwidth=8):
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
    loc = ((normLoc + 1) / 2) * mnist_size
    loc = T.cast(loc, 'int32')

    # img with size (batch, channels, height, width)
    img = T.reshape(img_batch, (batch_size, channels, mnist_size, mnist_size))
    # with size (batch, h, w, 1) after reshape

    zooms = []  # zooms of all the images in batch

    maxRadius = minRadius * (2 ** (depth - 1))  # radius of the largest zoom
    offset = maxRadius

    # zero-padding the batch to (batch, h + 2R, w + 2R, channels)
    img = T.concatenate((T.zeros((batch_size, channels, maxRadius, mnist_size)), img), axis=2)
    img = T.concatenate((img, T.zeros((batch_size, channels, maxRadius, mnist_size))), axis=2)
    img = T.concatenate((T.zeros((batch_size, channels, mnist_size + 2 * maxRadius, maxRadius)), img), axis=3)
    img = T.concatenate((img, T.zeros((batch_size, channels, mnist_size + 2 * maxRadius, maxRadius))), axis=3)
    img = T.cast(img, dtype=theano.config.floatX)

    for k in xrange(batch_size):
        imgZooms = []  # zoom for a single image

        # one_img with size (channels, 2R + size, 2R + size), channels=1 here
        one_img = img[k, :, :, :]

        for i in xrange(depth):
            # r = minR, 2 * minR, ..., (2^(depth - 1)) * minR
            r = minRadius * (2 ** i)

            d_raw = 2 * r  # patch size to be cropped

            loc_k = loc[k, :]  # location of the k-th glimpse, (2, )
            adjusted_loc = T.cast(offset + loc_k - r, 'int32')  # upper-left corner of the patch

            # one_img = T.reshape(one_img, (one_img.shape[0], one_img.shape[1]))

            # Get a zoom patch with size (d_raw, d_raw) from one_image
            # zoom = one_img[adjusted_loc[0]: (adjusted_loc[0] + d_raw),
            #        adjusted_loc[1]: (adjusted_loc[1] + d_raw)]
            zoom = one_img[:, adjusted_loc[0]: (adjusted_loc[0] + d_raw),
                   adjusted_loc[1]: (adjusted_loc[1] + d_raw)]

            if r < sensorBandwidth:  # bilinear-interpolation
                #  here, zoom is a 2D patch with size (2r, 2r)
                # zoom = T.swapaxes(zoom, 1, 2)
                # zoom = T.swapaxes(zoom, 0, 1)  # here, zoom with size (channel, height, width)
                zoom_reshape = T.reshape(zoom, (1, zoom.shape[0], zoom.shape[1], zoom.shape[2]))
                zoom_bandwidth = upsample.bilinear_upsampling(zoom_reshape,
                                                              ratio=(sensorBandwidth / r),
                                                              batch_size=1, num_input_channels=channels)
                # bandwith is with size (channel, height, width)
                zoom_bandwidth = T.reshape(zoom_bandwidth, (zoom_bandwidth.shape[1],
                                                            zoom_bandwidth.shape[2],
                                                            zoom_bandwidth.shape[3]))
            elif r > sensorBandwidth:
                # pooling operation will be down over the last 2 dimension
                # zoom = T.swapaxes(zoom, 1, 2)
                # zoom = T.swapaxes(zoom, 0, 1)  # here, zoom with size (channel, height, width)
                zoom_bandwidth = pool.pool_2d(input=zoom,
                                              ds=(r / sensorBandwidth,
                                                  r / sensorBandwidth),
                                              mode='average_inc_pad',
                                              ignore_border=True)
            else:
                zoom_bandwidth = zoom

            imgZooms.append(zoom_bandwidth)

        zooms.append(T.stack(imgZooms))

    # returned zooms is with size (batch, depth, channel, height, width)
    return T.stack(zooms)




""" Glimpse network
"""


def initial_W_b(rng, n_in, n_out):
    """ This function initializes the
        params for a single layer MLP

    :param n_in: input dimension
    :param n_out: output dimension
    """
    W_values = numpy.asarray(rng.uniform(
        low=-numpy.sqrt(6. / (n_in + n_out)),
        high=numpy.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out)
    ),
        dtype=theano.config.floatX
    )

    b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
    return W_values, b_values

""" Initialize parameters
"""
# 1-st set of params: W_hl, b_hl
W_values, b_values = initial_W_b(rng=numpy.random.RandomState(1234),
                                 n_in=2, n_out=hl_size)
W_hl = theano.shared(value=W_values, name='hl_net#W', borrow=True)
b_hl = theano.shared(value=b_values, name='hl_net#b', borrow=True)

# 2-nd set of params: W_hg, b_hg
W_values, b_values = initial_W_b(rng=numpy.random.RandomState(2345),
                                 n_in=totalSensorBandwidth,
                                 n_out=hg_size)
W_hg = theano.shared(value=W_values, name='hg_net#W', borrow=True)
b_hg = theano.shared(value=b_values, name='hg_net#b', borrow=True)


# 3-rd set of params: W_g, b_g
W_values, b_values = initial_W_b(rng=numpy.random.RandomState(3456),
                                 n_in=hl_size + hg_size,
                                 n_out=g_size)
W_g = theano.shared(value=W_values, name='g_net#W', borrow=True)
b_g = theano.shared(value=b_values, name='g_net#b', borrow=True)

# 4-th part, LSTMCore params: W, U
W_lstm = init_weights(shape=(g_size, 4 * cell_size),
                      name='LSTMCore#W')
U_lstm = theano.shared(
    value=numpy.concatenate((ortho_weight(ndim=cell_size),
                             ortho_weight(ndim=cell_size),
                             ortho_weight(ndim=cell_size),
                             ortho_weight(ndim=cell_size)),
                            axis=1), name='LSTMCore#U')
b_lstm = init_bias(size=4 * cell_size, name='LSTMCore#b')

# 5-th part set of params gl_out: W_hl_out, b_hl_out
W_values, b_values = initial_W_b(rng=numpy.random.RandomState(4567),
                                 n_in=cell_size,
                                 n_out=2)
W_hl_out = theano.shared(value=W_values, name='hl_out#W', borrow=True)
b_hl_out = theano.shared(value=b_values, name='hl_out#b', borrow=True)


def _slice(x, n, dim):
    if x.ndim == 3:
        return x[:, :, n * dim: (n + 1) * dim]
    return x[:, n * dim: (n + 1) * dim]

""" Write the output for one step
"""
l_tm1 = normLoc
x_t = img_batch
c_tm1 = T.alloc(floatX(0.), batch_size, cell_size)
h_tm1 = T.alloc(floatX(0.), batch_size, cell_size)


def _step(x_t, l_tm1, c_tm1, h_tm1):
    # 1-st part, hl_output with size (batch, hl_size)
    hl_output = T.nnet.relu(T.dot(l_tm1, W_hl) + b_hl)

    # 2-nd part, hg_net
    glimpse_input = glimpseSensor(x_t, l_tm1)
    hg_input = T.reshape(glimpse_input, (glimpse_input.shape[0],
                                         totalSensorBandwidth))
    hg_output = T.nnet.relu(T.dot(hg_input, W_hg) + b_hg)

    # 3-rd part, g_net with output size (batch, g_size)
    g_output = T.nnet.relu(T.dot(T.concatenate((hl_output, hg_output),
                                               axis=1), W_g) + b_g)

    preact = T.dot(g_output, W_lstm) + T.dot(h_tm1, U_lstm) + b_lstm

    i = T.nnet.sigmoid(_slice(preact, 0, cell_size))
    f = T.nnet.sigmoid(_slice(preact, 1, cell_size))
    o = T.nnet.sigmoid(_slice(preact, 2, cell_size))
    c_tilde = T.tanh(_slice(preact, 3, cell_size))

    c_t = f * c_tm1 + i * c_tilde
    h_t = o * T.tanh(c_t)
    l_t = T.tanh(T.dot(h_t, W_hl_out) + b_hl_out)

    return l_t, c_t, h_t

l_t, c_t, h_t = _step(x_t, l_tm1, c_tm1, h_tm1)

srng = RandomStreams(seed=234)

sampled_l_t = srng.normal(size=l_t.shape, avg=l_t, std=loc_sd)


fn_step = theano.function(inputs=[img_batch, normLoc],
                          outputs=[c_t, h_t, sampled_l_t])
c, h, l = fn_step(images, locs)


print type(c)
print c.shape

print type(h)
print h.shape

print type(l)
print l.shape
print l[0: 5, :]


"""
fn_g = theano.function(inputs=[img_batch, normLoc],
                       outputs=[g_output])
g_out = fn_g(images, locs)[0]
print type(g_out)
print g_out.shape
"""


"""
fn_reshape = theano.function(inputs=[img_batch, normLoc],
                             outputs=[glimpseLayer.zooms])

zoom_reshape = fn_reshape(images, locs)[0]

f = h5py.File('zoom.h5')
f['zoom'] = zoom_reshape
f.close()

print type(zoom_reshape)
print zoom_reshape.shape
"""




