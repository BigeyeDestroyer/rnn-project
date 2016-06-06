import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import sys
sys.path.append('..')
from common.utils import *


class SoftmaxLayer(object):
    def __init__(self, X, shape, layer_id, epsilon=1.0e-15):
        """
        Softmax layer

        Parameters
        ----------
        :type layer_id: str
        :param layer_id: id of this layer

        :type shape: tuple
        :param shape: (in_size, out_size) where
                      in_size is the input dimension
                      out_size is the final output size
        :type X: a 2D variable, of size (n_samples, in_size)
        :param X: model inputs
        """
        prefix = 'Softmax' + layer_id
        self.in_size, self.out_size = shape
        self.W = init_weights(shape=(self.in_size, self.out_size),
                              name=prefix + '#W')
        self.b = init_bias(size=self.out_size,
                           name=prefix + '#b')

        self.X = X
        self.params = [self.W, self.b]

        self.epsilon = epsilon

        """
        def _step(x_t):
            o_t = T.nnet.softmax(T.dot(x_t, self.W) + self.b)
            return o_t

        o, updates = theano.scan(fn=_step, sequences=[self.X])
        """

        # activation of size (n_samples, hid_size)
        self.p_y_given_x = T.nnet.softmax(T.dot(self.X, self.W) + self.b)
        self.y_pred_prob = T.max(self.p_y_given_x, axis=1)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        p_y_given_x = T.clip(self.p_y_given_x, self.epsilon, 1 - self.epsilon)
        return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
