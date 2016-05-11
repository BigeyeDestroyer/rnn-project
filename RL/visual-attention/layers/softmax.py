import theano.tensor as T
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
                 or a 3D variable, of size (t, n_samples, in_size)
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
        if self.X.ndim == 3:
            # activation of size (t * n_samples, out_size)
            activation = T.dot(self.X, self.W) + self.b
            activation = T.reshape(activation, (activation.shape[0] * activation.shape[1],
                                                activation.shape[2]))
            self.p_y_given_x = T.nnet.softmax(activation)

        else:
            # activation of size (n_samples, out_size)
            activation = T.dot(self.X, self.W) + self.b
            self.p_y_given_x = T.nnet.softmax(activation)

        # self.y_pred_prob = T.max(self.p_y_given_x, axis=1)
        # if X.ndim == 2, 'y_pred' with size (n_samples, )
        # if X.ndim == 3, 'y_pred' with size (t * n_samples, )
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y, mask=None):
        """
        This function computes the log-likelihood

        Parameters
        ----------
        :param y: with size (n_samples, )     when X.ndim == 2
                  with size (t * n_samples, ) when X.ndim == 3

        :param mask: None when X.ndim == 2
                     with size (t * n_samples, ) when X.ndim == 3
        """
        p_y_given_x = T.clip(self.p_y_given_x, self.epsilon, 1 - self.epsilon)
        if mask is not None:
            # when X.ndim = 3
            cost = -T.sum(T.log(p_y_given_x)[T.arange(y.shape[0]), y] * mask) / T.sum(mask)

        else:
            # when X.ndim = 2
            cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

        return cost

    def errors(self, y, mask=None):
        """
        This function computes the error

        Parameters
        ----------
        :param y: with size (n_samples, )     when X.ndim == 2
                  with size (t * n_samples, ) when X.ndim == 3

        :param mask: None when X.ndim == 2
                     with size (t * n_samples, ) when X.ndim == 3
        """
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            if mask is not None:
                error = T.sum(T.neq(self.y_pred, y) * mask) / T.sum(mask)
            else:
                error = T.mean(T.neq(self.y_pred, y))

            return error
        else:
            raise NotImplementedError()
