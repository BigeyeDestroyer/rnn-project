from embeddings import *
from lstm import *
from gru import *
from basic import *
from softmax import *
import h5py
import sys
sys.path.append('..')
from optimizers.optimizers import *


class RNN(object):
    def __init__(self, n_words, in_size, out_size, hidden_size,
                 cell='lstm', optimizer='adam', p=0.5):
        """
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
            else:
                hidden_layer = BasicLayer(rng, str(i + 1), shape, layer_input, self.maskX,
                                          self.use_noise, self.drop_rate)

            self.layers.append(hidden_layer)
            self.params += hidden_layer.params

        # output layer, we just average over all the output
        # here the layer_input is with size (t, n, out_size)
        layer_input = hidden_layer.activation  # output of the last lstm
        layer_input = (layer_input * self.maskX[:, :, None]).sum(axis=0)
        layer_input = layer_input / self.maskX.sum(axis=0)[:, None]  # (n_samples, hidden_size)

        output_layer = SoftmaxLayer(X=layer_input, shape=(hidden_layer.hid_size, self.out_size),
                                    layer_id=str(len(self.hidden_size) + 1), epsilon=self.epsilon)
        self.layers.append(output_layer)
        self.params += output_layer.params

    def train_test_funcs(self):
        # pred is of size (n_samples, out_size)


        # pred = self.layers[len(self.layers) - 1].activation
        # y_pred_prob = pred.max(axis=1)
        # y_pred = pred.argmax(axis=1)  # the predicted labels

        self.y = T.vector('y', dtype='int64')

        cost = self.layers[-1].negative_log_likelihood(self.y)
        error = self.layers[-1].errors(self.y)

        y_pred_prob = self.layers[-1].y_pred_prob
        y_pred = self.layers[-1].y_pred

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

        self.pred_prob = theano.function(inputs=[self.X, self.maskX],
                                         givens={self.use_noise: numpy.cast['int32'](0)},
                                         outputs=y_pred_prob)

        self.pred = theano.function(inputs=[self.X, self.maskX],
                                    givens={self.use_noise: numpy.cast['int32'](0)},
                                    outputs=y_pred)

        self.error = theano.function(inputs=[self.X, self.maskX, self.y],
                                     givens={self.use_noise: numpy.cast['int32'](0)},
                                     outputs=error)

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









