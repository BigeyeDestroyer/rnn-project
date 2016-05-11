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


class glimpseSensor(object):
    def __init__(self, layer_id, img, normLoc, mnist_size=28, channels=1, depth=3, minRadius=4, sensorBandwidth=8):
        """ Recurrent Attention Model from
        "Recurrent Models of Visual Attention" (Mnih + 2014)

        Parameters
        ----------
        :type layer_id: str
        :param layer_id: id of this layer

        :type img: a 2D variable, each row an mnist image
        :param img: model inputs

        :type normLoc: ndarray with size batch_size x 2
        :param normLoc: normalized initial locations of glimpses

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
        prefix = 'glimpseSensor' + layer_id
        self.channels = channels
        self.mnist_size = mnist_size
        self.minRadius = minRadius
        self.sensorBandwidth = sensorBandwidth

        # from [-1.0, 1.0] -> [0, 28]
        loc = ((normLoc + 1) / 2) * self.mnist_size
        loc = loc.astype(numpy.int32)

        batch_size = img.shape[0]
        img = T.reshape(img, (batch_size, mnist_size, mnist_size, channels))
        self.img = img  # with size (batch, h, w, 1)

        # process each image individually
        zooms = []
        for k in xrange(batch_size):
            imgZooms = []

            maxRadius = minRadius * (2 ** (depth - 1))  # radius of the largest zoom
            offset = maxRadius

            # pad image with zeros
            # original size : (mnist_size, mnist_size, 1)
            # padded size   : (mnist_size + 2 * maxRadius, mnist_size + 2 * maxRadius, 1)
            one_img = T.zeros([mnist_size + 2 * maxRadius,
                               mnist_size + 2 * maxRadius, 1],
                              dtype=theano.config.floatX)
            one_img[maxRadius: (mnist_size + maxRadius),
            maxRadius: (mnist_size + maxRadius), :] = self.img[k, :, :, :]

            for i in xrange(depth):
                # r = minR, 2* minR, ..., (2^(depth - 1)) * minR
                r = int(minRadius * (2 ** i))

                d_raw = 2 * r
                


