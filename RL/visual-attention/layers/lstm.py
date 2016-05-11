import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import sys
sys.path.append('..')
from common.utils import *
import numpy
import theano


class LSTMLayer(object):
    def __init__(self, rng, layer_id, shape, X, mask,
                 use_noise=1, p=0.5):
        """
        Basic LSTM initialization function with dropout

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
        prefix = 'LSTM' + layer_id
        self.in_size, self.hid_size = shape

        # weights for input, in the order W_i, W_f, W_o and W_c
        self.W = init_weights(shape=(self.in_size, 4 * self.hid_size),
                              name=prefix + '#W')
        # weights for hidden states
        self.U = theano.shared(value=numpy.concatenate((ortho_weight(ndim=self.hid_size),
                                                        ortho_weight(ndim=self.hid_size),
                                                        ortho_weight(ndim=self.hid_size),
                                                        ortho_weight(ndim=self.hid_size)),
                                                       axis=1),
                               name=prefix + '#U')

        # self.U = init_weights(shape=(self.hid_size, 4 * self.hid_size),
        #                       name=prefix + '#U')
        # bias
        self.b = init_bias(size=4 * self.hid_size, name=prefix + '#b')

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

        def _step(x_t, m_t, c_tm1, h_tm1):
            """
            This function computes one step evolution in LSTM

            Parameters
            ----------
            :type m_t: (n_samples, )
            :param m_t: mask

            :type x_t: (n_samples, in_size)
            :param x_t: input at time t

            :type c_tm1: (n_samples, hid_size)
            :param c_tm1: cell state at time (t - 1)

            :type h_tm1: (n_samples, hid_size)
            :param h_tm1: hidden state at time (t - 1)
            """
            preact = T.dot(x_t, self.W) + T.dot(h_tm1, self.U) + self.b

            i = T.nnet.sigmoid(_slice(preact, 0, self.hid_size))
            f = T.nnet.sigmoid(_slice(preact, 1, self.hid_size))
            o = T.nnet.sigmoid(_slice(preact, 2, self.hid_size))
            c_tilde = T.tanh(_slice(preact, 3, self.hid_size))

            c_t = f * c_tm1 + i * c_tilde
            # consider the mask
            c_t = m_t[:, None] * c_t + (1. - m_t)[:, None] * c_tm1

            h_t = o * T.tanh(c_t)
            # consider the mask
            h_t = m_t[:, None] * h_t + (1. - m_t)[:, None] * h_tm1

            return c_t, h_t

        [c, h], _ = theano.scan(fn=_step,
                                sequences=[self.X, self.mask],
                                outputs_info=[T.alloc(floatX(0.),
                                                      n_samples,
                                                      self.hid_size),
                                              T.alloc(floatX(0.),
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

        self.params = [self.W, self.U, self.b]


class BdLSTMLayer(object):
    # Bidirectional LSTM layer
    def __init__(self, rng, layer_id, shape, X, mask,
                 use_noise=1, batch_size=1, p=0.5):
        fwd = LSTMLayer(rng, '_fwd_' + layer_id, shape, X, mask, use_noise, batch_size, p)
        bwd = LSTMLayer(rng, '_bwd_' + layer_id, shape, X[::-1], mask[::-1], use_noise, batch_size, p)
        self.params = fwd.params + bwd.params
        # suppose fwd.activation is of size    (t, n_samples, hid_size)
        # thus, bwd.activation is also of size (t, n_samples, hid_size)
        # so after concatenation, we have size (t, n_samples, 2 * hid_size)
        self.activation = T.concatenate([fwd.activation, bwd.activation[::-1]], axis=2)



