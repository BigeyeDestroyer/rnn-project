import theano.tensor as T
import sys
sys.path.append('..')
import theano
import theano.tensor.signal.pool as pool
import theano.tensor.nnet.abstract_conv as upsample


class glimpseSensor(object):
    def __init__(self, img_batch, normLoc, batch_size=16, mnist_size=28, channels=1, depth=3, minRadius=4, sensorBandwidth=8):
        """ Recurrent Attention Model from
        "Recurrent Models of Visual Attention" (Mnih + 2014)

        Parameters
        ----------
        :type layer_id: str
        :param layer_id: id of this layer

        :type img_batch: a 2D variable, each row an mnist image
        :param img_batch: model inputs

        :type normLoc: variable with size (batch_size x 2)
        :param normLoc: model inputs

        :type batch_size: int
        :param batch_size: batch size

        :type mnist_size: int
        :param mnist_size: length of the mnist square (usually 28)

        :type channels: int
        :param channels: channels of mnist (usually 1)

        :type depth: int
        :param depth: channels of zoom (3 in this paper)

        :type minRadius: int
        :param minRadius: minimum radius of the glimpse

        :type sensorBandwidth: int
        :param sensorBandwidth: length of the glimpse square

        :return self.zooms: (batch, depth, channel, height, width)
        """
        self.batch_size = batch_size
        self.mnist_size = mnist_size
        self.channels = channels
        self.depth = depth
        self.minRadius = minRadius
        self.sensorBandwidth = sensorBandwidth

        # from [-1.0, 1.0] -> [0, 28]
        loc = ((normLoc + 1) / 2) * mnist_size
        loc = T.cast(loc, 'int32')

        # img with size (batch, channels, height, width)
        img = T.reshape(img_batch, (batch_size, channels, mnist_size, mnist_size))
        self.img = img  # with size (batch, 1, h, w)

        zooms = []  # zooms of all the images in batch

        maxRadius = minRadius * (2 ** (depth - 1))  # radius of the largest zoom
        offset = maxRadius

        # zero-padding the batch to (batch, channels, h + 2R, w + 2R)
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
                # zoom with size (channels, 2 * r, 2 * r)
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

        # returned self.zooms is with size (batch, depth, channel, height, width)
        self.zooms = T.stack(zooms)





