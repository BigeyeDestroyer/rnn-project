import sys
sys.path.append('..')
from common.utils import *

class EmbeddingLayer(object):
    def __init__(self, layer_id,
                 shape, X):
        """
        Embedding layer for rnn model

        Parameters
        ----------
        :param rng: can be generated as numpy.random.seed(123)

        :type layer_id: str
        :param layer_id: id of this layer

        :type shape: tuple
        :param shape: (n_words, in_size) where
                      n_words is the vocabulary size
                      in_size is the embedding dimension

        :type X: a 2D matrix
        :param X: model inputs to be embedded

        :type mask: theano variable
        :param mask: model inputs

        :type batch_size: int
        :param batch_size: mini-batch's size
        """
        prefix = 'Embedding' + layer_id
        self.n_words, self.in_size = shape

        # weights for embedding, the only parameters
        self.W = init_weights(shape=(self.n_words, self.in_size),
                              name=prefix + '#W')

        self.X = X
        self.n_timesteps = X.shape[0]
        self.n_samples = X.shape[1]

        self.activation = self.W[self.X.flatten()].reshape([self.n_timesteps,
                                                            self.n_samples,
                                                            self.in_size])
        self.params = [self.W]
