"""
Below are the params needed for glimpseSensor:

minRadius = 4
sensorBandwidth = 8  # fixed resolution of sensor
sensorArea = sensorBandwidth ** 2
depth = 3  # channels of zoom
channels = 1  # grayscale image
totalSensorBandwidth = depth * sensorBandwidth * \
                       sensorBandwidth * channels

mnist_size = 28

"""
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import sys
sys.path.append('..')
from common.utils import *
import numpy
import theano
import theano.tensor.signal.pool as pool
import theano.tensor.nnet.abstract_conv as upsample


class glimpseSensor(object):
    def __init__(self, img, normLoc, batch_size, mnist_size=28, channels=1, depth=3, minRadius=4, sensorBandwidth=8):
        """ Recurrent Attention Model from
        "Recurrent Models of Visual Attention" (Mnih + 2014)

        Parameters
        ----------
        :type layer_id: str
        :param layer_id: id of this layer

        :type img: a 2D variable, each row an mnist image
        :param img: model inputs

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
        """

        self.mnist_size = mnist_size
        self.channels = channels
        self.depth = depth
        self.minRadius = minRadius
        self.sensorBandwidth = sensorBandwidth

        # from [-1.0, 1.0] -> [0, 28]
        loc = ((normLoc + 1) / 2) * self.mnist_size
        loc = T.cast(loc, 'int32')

        img = T.reshape(img, (batch_size, mnist_size, mnist_size, channels))
        self.img = img  # with size (batch, h, w, 1)


        zooms = []
        # process each image individually
        for k in xrange(batch_size):
            imgZooms = []

            maxRadius = minRadius * (2 ** (depth - 1))  # radius of the largest zoom
            offset = maxRadius

            # pad image with zeros
            # original size : (mnist_size, mnist_size, 1)
            # padded size   : (mnist_size + 2 * maxRadius, mnist_size + 2 * maxRadius, 1)
            img_up = T.concatenate((T.zeros((maxRadius, mnist_size, 1)), self.img[k, :, :, :]), axis=0)
            img_down = T.concatenate((img_up, T.zeros((maxRadius, mnist_size, 1))), axis=0)
            img_left = T.concatenate((T.zeros((mnist_size + 2 * maxRadius, maxRadius, 1)), img_down), axis=1)
            one_img = T.concatenate((img_left, T.zeros((mnist_size + 2 * maxRadius, maxRadius, 1))), axis=1)

            one_img = T.cast(one_img, dtype=theano.config.floatX)

            for i in xrange(depth):
                # r = minR, 2* minR, ..., (2^(depth - 1)) * minR
                r = int(minRadius * (2 ** i))

                d_raw = 2 * r  # patch size to be cropped
                d = [d_raw, d_raw]  # the patch size to be cropped

                loc_k = loc[k, :]  # location of the k-th glimpse, (2, )
                adjusted_loc = T.cast(offset + loc_k - r, 'int32')  # upper-left corner of the patch

                # reshape makes a tensor with size (n, n, 1) to (n, n)
                one_img_2D = T.reshape(one_img, (one_img.shape[0], one_img.shape[1]))
                zoom = one_img_2D[adjusted_loc[0]: (adjusted_loc[0] + d),
                       adjusted_loc[1]: (adjusted_loc[1] + d)]
                if r < sensorBandwidth:  # bilinear-interpolation
                    # here, zoom is a 2D patch
                    zoom_reshape = T.reshape(zoom, (1, 1, zoom.shape[0], zoom.shape[1]))
                    zoom_bandwidth = upsample.bilinear_upsampling(zoom_reshape,
                                                                  ratio=(sensorBandwidth / r),
                                                                  batch_size=1, num_input_channels=1)
                    zoom_bandwidth = T.reshape(zoom_bandwidth, (zoom_bandwidth.shape[2],
                                                                zoom_bandwidth.shape[3]))
                elif r > sensorBandwidth:
                    zoom_bandwidth = pool.pool_2d(input=zoom,
                                                  ds=(r / sensorBandwidth,
                                                      r / sensorBandwidth),
                                                  mode='average_inc_pad',
                                                  ignore_border=True)
                imgZooms.append(zoom_bandwidth)  # zoom for a single image

            zooms.append(T.stack(imgZooms))  # stack -> (depth, width, height)

        self.zooms = T.stack(zooms)  # stack -> (batch, depth, width, height)






