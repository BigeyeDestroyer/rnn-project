import theano.tensor as T
import theano
import numpy
import h5py
from glimpseSensor import glimpseSensor
from HiddenLayer import HiddenLayer


class Glimpse(object):
    def __init__(self, glimpse_input, normLoc, channels=1, depth=3,
                 sensorBandwidth=8, hg_size=128, hl_size=128, g_size=256):
        """
        :type glimpse_input: 5D variable with size
                            (batch, depth, channels, height, width)
        :param glimpse_input: glimpse input from the sensor

        :type normLoc: 2D variable with size (batch, 2)
        :param normLoc: locations of the sensor

        :type channels: int
        :param channels: channels of the input images

        :type depth: int
        :param depth: number of channels of the zooms

        :type sensorBandwidth: int
        :param sensorBandwidth: radius of the sensor

        :type hg_size: int
        :param hg_size: dimension of the hg_net

        :type hl_size: int
        :param hl_size: dimension of the hl_net

        :type g_size: int
        :param g_size: dimension of the final glimpse net

        :return self.output: with size (batch, g_size), g_size = 256
        """

        self.channels = channels
        self.depth = depth
        self.sensorBandwidth = sensorBandwidth
        self.hg_size = hg_size
        self.hl_size = hl_size
        self.g_size = g_size

        # location net to process the input location with size (batch, 2)
        hl_net = HiddenLayer(rng=numpy.random.RandomState(1234), prefix='hl_net',
                             input=normLoc, n_in=2, n_out=hl_size,
                             activiation=T.nnet.relu)

        # glimpse_input net to process the glimpseSensor result
        # with size (batch, depth * channels * (2 * bandwidth) * (2 * bandwidth))
        totalSensorBandwidth = depth * channels * ((2 * sensorBandwidth) ** 2)
        hg_input = T.reshape(glimpse_input, (glimpse_input.shape[0],
                                             totalSensorBandwidth))

        hg_net = HiddenLayer(rng=numpy.random.RandomState(2345), prefix='hg_net',
                             input=hg_input, n_in=totalSensorBandwidth, n_out=hg_size,
                             activiation=T.nnet.relu)

        # concatenate hl_net's output and hg_net's output
        g_net = HiddenLayer(rng=numpy.random.RandomState(3456), prefix='g_net',
                            input=T.concatenate((hl_net.output, hg_net.output), axis=1),
                            n_in=hl_size + hg_size, n_out=g_size,
                            activiation=T.nnet.relu)

        # The glimpse output, with size of (batch, 256)
        self.output = g_net.output


