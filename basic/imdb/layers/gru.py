import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils import *


class GRULayer(object):
    def __init__(self, rng, layer_id, shape, X, mask,
                 use_noise=1, p=0.5):
        """
        Basic GRU initialization function with dropout

        Parameters
        ----------
        :param rng: can be generated as numpy.random.seed(123)

        :type layer_id: str
        :param layer_id: id of this layer

        :type shape: tuple
        :param shape: (in_size, out_size) where
                      in_size is the input dimension
                      out_size is the hidden units' dimension

        :type X: a 3D or 2D variable, mostly a 3D one
        :param X: model inputs

        :type mask: theano variable
        :param mask: model inputs

        :type use_noise: theano variable
        :param use_noise: whether dropout is random

        :type p: float
        :param p: dropout ratio
        """

        prefix = 'GRU' + layer_id
        self.in_size, self.hid_size = shape

        # update gate related parameters
        self.Wz = init_weights(shape=(self.in_size, self.hid_size), name=prefix + '#Wz')
        self.Uz = init_weights(shape=(self.hid_size, self.hid_size), name=prefix + '#Uz')
        self.bz = init_bias(size=self.hid_size, name=prefix + '#bz')

        # reset gate related parameters
        self.Wr = init_weights(shape=(self.in_size, self.hid_size), name=prefix + '#Wr')
        self.Ur = init_weights(shape=(self.hid_size, self.hid_size), name=prefix + '#Ur')
        self.br = init_bias(size=self.hid_size, name=prefix + '#br')

        # h_tiled related parameters
        self.Wh = init_weights(shape=(self.in_size, self.hid_size), name=prefix + '#Wh')
        self.Uh = init_weights(shape=(self.hid_size, self.hid_size), name=prefix + '#Uh')
        self.bh = init_bias(size=self.hid_size, name=prefix + '#bh')

        self.X = X
        self.mask = mask

        nsteps = X.shape[0]
        if X.ndim == 3:
            n_samples = X.shape[1]
        else:
            n_samples = 1

        assert mask is not None

        def _slice(x, n, dim):
            if x.ndim == 3:
                return x[:, :, n * dim: (n + 1) * dim]
            return x[:, n * dim: (n + 1) * dim]

        def _step(x_t, m_t, h_tm1):
            """
            This function computes one step evolution in GRU

            Parameters
            ----------
            :type x_t: (n_samples, in_size)
            :param x_t: input at time t

            :type m_t: (n_samples, )
            :param m_t: mask

            :type h_tm1: (n_samples, hid_size)
            :param h_tm1: hidden state at time (t - 1)
            """
            z_t = T.nnet.sigmoid(T.dot(x_t, self.Wz) + T.dot(h_tm1, self.Uz) + self.bz)
            r_t = T.nnet.sigmoid(T.dot(x_t, self.Wr) + T.dot(h_tm1, self.Ur) + self.br)
            h_tiled = T.tanh(T.dot(x_t, self.Wh) + T.dot(r_t * h_tm1, self.Uh) + self.bh)

            h_t = (1. - z_t) * h_tm1 + z_t * h_tiled
            # consider mask
            h_t = m_t[:, None] * h_t + (1. - m_t)[:, None] * h_tm1
            return h_t

        h, updates = theano.scan(fn=_step, sequences=[self.X, self.mask],
                                 outputs_info=[T.alloc(floatX(0.),
                                                       n_samples,
                                                       self.hid_size)])
        # h here is of size (t, n_samples, hid_size)
        if p > 0:
            trng = RandomStreams(rng.randint(999999))
            drop_mask = trng.binomial(size=h.shape, n=1,
                                      p=(1 - p), dtype=theano.config.floatX)
            self.activation = T.switch(T.eq(use_noise, 1), h * drop_mask, h * (1 - p))
        else:
            self.activation = h

        self.params = [self.Wz, self.Uz, self.bz,
                       self.Wr, self.Ur, self.br,
                       self.Wh, self.Uh, self.bh]