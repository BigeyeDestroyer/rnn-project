import numpy
import theano
import theano.tensor as T
from utils import *


class LogisticLayer(object):
    def __init__(self, layer_id, shape, X):
        """
        Logistic regression layer

        Parameters
        ----------
        :type layer_id: str
        :param layer_id: id of this layer

        :type shape: tuple
        :param shape: (in_size, out_size) where
                      in_size is the input dimension
                      out_size is the final output size
        :type X: a 3D or 2D variable, mostly a 3D one
                 and of size (t, n_samples, in_size)
        :param X: model inputs
        """
        prefix = 'Logistic' + layer_id
        self.in_size, self.out_size = shape
        self.W = init_weights(shape=shape, name=prefix + '#W')
        self.b = init_bias(size=self.out_size, name=prefix + '#b')

        # X may be of size (t, n_samples, in_size)
        self.X = X

        self.activation = T.nnet.sigmoid(T.dot(self.X, self.W) + self.b)
        self.params = [self.W, self.b]

