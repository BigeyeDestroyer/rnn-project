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
        :type layer_id: str
        :param layer_id: id of this layer

        :type shape: tuple
        :param shape: (n_words, in_size) where
                      n_words is the vocabulary size
                      in_size is the embedding dimension

        :type X: a 2D matrix with size (n_timestamps, n_samples)
        :param X: model inputs to be embedded

        :type batch_size: int
        :param batch_size: mini-batch's size
        """
        prefix = 'Embedding' + layer_id
        self.n_words, self.in_size = shape

        # weights for embedding, the only parameters
        self.W = init_weights(shape=(self.n_words, self.in_size),
                              name=prefix + '#W')

        self.params = [self.W]

        # Compute the embedded samples
        self.n_timesteps = X.shape[0]
        self.n_samples = X.shape[1]

        self.activation = self.W[X.flatten()].reshape([self.n_timesteps,
                                                       self.n_samples,
                                                       self.in_size])

