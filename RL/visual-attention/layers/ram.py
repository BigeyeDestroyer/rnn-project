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

        :param batch_size:
        :param mnist_size:
        :param channels:
        :param depth:
        :param minRadius:
        :param sensorBandwidth:


        :param hg_size:
        :param hl_size:
        :param g_size:

        :param cell_size:
        :param cell_out_size:
        :param n_classes:

        :param loc_sd:
        :param glimpses:
        :param numpy_rng:
        :return:
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
        self.theano_rng = RandomStreams(numpy_rng)
        self.totalSensorBandwidth = depth * channels * ((2 * sensorBandwidth) ** 2)

        def initial_W_b(rng, n_in, n_out):
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
        W_values, b_values = initial_W_b(rng=numpy_rng,
                                         n_in=2, n_out=self.hl_size)
        self.W_hl = theano.shared(value=W_values, name='hl_net#W', borrow=True)
        self.b_hl = theano.shared(value=b_values, name='hl_net#b', borrow=True)

        # 2-nd set of params: W_hg, b_hg, params for the hg_net
        W_values, b_values = initial_W_b(rng=numpy_rng,
                                         n_in=self.totalSensorBandwidth,
                                         n_out=self.hg_size)
        self.W_hg = theano.shared(value=W_values, name='hg_net#W', borrow=True)
        self.b_hg = theano.shared(value=b_values, name='hg_net#b', borrow=True)

        # 3-rd set of params: W_g, b_g, params for the h_g net
        W_values, b_values = initial_W_b(rng=numpy_rng,
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
        W_values, b_values = initial_W_b(rng=numpy_rng,
                                         n_in=self.cell_size,
                                         n_out=2)
        self.W_hl_out = theano.shared(value=W_values, name='hl_out#W', borrow=True)
        self.b_hl_out = theano.shared(value=b_values, name='hl_out#b', borrow=True)

        # 6-th set of params: W_ha_out, b_ha_out, params for the ha_out net
        W_values, b_values = initial_W_b(rng=numpy_rng,
                                         n_in=self.cell_size,
                                         n_out=self.n_classes)
        self.W_ha_out = theano.shared(value=W_values, name='ha_out#W', borrow=True)
        self.b_ha_out = theano.shared(value=b_values, name='ha_out#b', borrow=True)

