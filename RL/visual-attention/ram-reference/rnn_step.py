import h5py
import sys
sys.path.append('..')
from layers.embeddings import *
from layers.gru import *
from layers.lstm import *
from layers.softmax import *
from optimizers.optimizers import *


class RNN(object):
    def __init__(self, n_words, in_size, out_size, hidden_size,
                 cell='gru', optimizer='rmsprop', p=0.5):
        """
        This rnn is outputs at each time step, and
        calculate cost, errors with respect to each
        time step output.

        Parameters
        ----------
        :type n_words: int
        :param n_words: vocabulary size

        :type in_size: int
        :param in_size: the projection dimension

        :type out_size: int
        :param out_size: output dimension, always
                         the number of classes

        :type hidden_size: list of ints
        :param hidden_size: indicates the hidden dimensions
                            of the stacked RNNs

        :type cell: str
        :param cell: 'LSTM' or 'GRU'

        :type: optimizer: str
        :param optimizer: the optimizers

        :type p: float
        :param p: dropout ratio
        """
        # X is of size (n_timesteps, n_samples)
        self.X = T.matrix('X', dtype='int64')
        self.n_words = n_words
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.cell = cell
        self.drop_rate = p
        self.use_noise = T.iscalar('use_noise')  # whether dropout is random
        self.batch_size = T.iscalar('batch_size')  # for mini-batch
        self.maskX = T.matrix('maskX', dtype=theano.config.floatX)
        self.maskY = T.matrix('maskY')
        self.optimizer = optimizer
        self.layers = []
        self.params = []
        self.epsilon = 1.0e-15  # for clipping in prediction step
        self.define_layers()
        self.train_test_funcs()

    def define_layers(self):

        rng = numpy.random.RandomState(1234)

        # embedding layer
        layer_input = self.X
        shape = (self.n_words, self.in_size)
        embed_layer = EmbeddingLayer(str(0), shape, layer_input)

        self.layers.append(embed_layer)
        self.params += embed_layer.params

        # hidden layers
        for i in xrange(len(self.hidden_size)):
            layer_input = self.layers[i].activation
            if i == 0:
                shape = (self.in_size, self.hidden_size[0])
            else:
                shape = (self.hidden_size[i - 1], self.hidden_size[i])

            if self.cell == 'gru':
                hidden_layer = GRULayer(rng, str(i + 1), shape, layer_input, self.maskX,
                                        self.use_noise, self.drop_rate)
            elif self.cell == 'lstm':
                hidden_layer = LSTMLayer(rng, str(i + 1), shape, layer_input, self.maskX,
                                         self.use_noise, self.drop_rate)

            self.layers.append(hidden_layer)
            self.params += hidden_layer.params

        # output layer, we just average over time for the current task
        # thus, the activation is (n_samples, out_size)
        layer_input = hidden_layer.activation  # output of the last lstm
        layer_input = (layer_input * self.maskX[:, :, None])

        output_layer = SoftmaxLayer(X=layer_input, shape=(hidden_layer.hid_size, self.out_size),
                                    layer_id=str(len(self.hidden_size) + 1), epsilon=self.epsilon)
        self.layers.append(output_layer)
        self.params += output_layer.params

    def bpc(self, y, mask):
        """
        This function computes the cost and bpc

        Parameters
        ----------
        :param y: with size (t * n_samples, )

        :param mask: with size (t * n_samples, )
        """
        # p_y_given_x is with size (t * n_samples, out_size)
        p_y_given_x = self.layers[-1].p_y_given_x
        p_y_given_x = T.clip(p_y_given_x, self.epsilon, 1 - self.epsilon)

        # compute bpc
        bpc = -T.sum(T.log2(p_y_given_x)[T.arange(y.shape[0]), y] * mask) / T.sum(mask)

        return bpc

    def train_test_funcs(self):
        # self.y is of size (t, n_samples)
        self.y = T.matrix('y', dtype='int64')

        t = self.y.shape[0]
        n_samples = self.y.shape[1]

        y_flat = T.reshape(self.y, (t * n_samples, ))
        mask_flat = T.reshape(self.maskX, (t * n_samples, ))

        cost = self.layers[-1].negative_log_likelihood(y_flat, mask_flat)
        error = self.layers[-1].errors(y_flat, mask_flat)
        bpc = self.bpc(y_flat, mask_flat)

        y_pred = self.layers[-1].y_pred
        y_pred = T.reshape(y_pred, (t, n_samples))

        gparams = []
        for param in self.params:
            gparam = T.clip(T.grad(cost, param), -10, 10)
            gparams.append(gparam)

        lr = T.scalar('lr')
        # eval(): string to function
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, lr)

        self.train = theano.function(inputs=[self.X, self.maskX, self.y, lr],
                                     givens={self.use_noise: numpy.cast['int32'](1)},
                                     outputs=cost, updates=updates)

        self.error = theano.function(inputs=[self.X, self.maskX, self.y],
                                     givens={self.use_noise: numpy.cast['int32'](0)},
                                     outputs=error)

        self.bpc = theano.function(inputs=[self.X, self.maskX, self.y],
                                   givens={self.use_noise: numpy.cast['int32'](0)},
                                   outputs=bpc)
        self.pred = theano.function(inputs=[self.X, self.maskX, self.y],
                                   givens={self.use_noise: numpy.cast['int32'](0)},
                                   outputs=y_pred)

    def save_to_file(self, file_name, file_index=None):
        """
        This function stores the trained params to '*.h5' file

        Parameters
        ----------
        :type file_dir: str
        :param file_dir: the directory with name to store trained parameters

        :type file_index: str, generated as str(1)
        :param file_index: if parameters here are snapshot,
                           then we need to add index to file name
        """
        if file_index is not None:
            file_name = file_name[:-3] + str(file_index) + '.h5'

        f = h5py.File(file_name)
        for p in self.params:
            f[p.name] = p.get_value()
        f.close()









