import theano
import theano.tensor as T
import numpy
from common.utils import *


class ControllerFeedforward(object):
    def __init__(self, X, read_input, input_size=8, output_size=8,
                 mem_size=128, mem_width=20, layer_sizes=[100]):
        """
        :type X: theano tensor, with size (batch, input_size)
        :param X: the input from outside environment

        :type read_input: theano tensor, with size (batch, mem_width)
        :param read_input: the input from memory matrix

        :type input_size: int
        :param input_size: the input size of outside input

        :type output_size: int
        :param output_size: the output size of outside output

        :type mem_size: int
        :param mem_size: rows of the memory matrix

        :type mem_width: int
        :param mem_width: length of each row in the memory matrix

        :type layer_sizes: list of ints
        :param layer_sizes: hidden sizes of the controller
        """
        self.X = X
        self.read_input = read_input

        self.input_size = input_size
        self.output_size = output_size
        self.mem_size = mem_size
        self.mem_width = mem_width
        self.layer_sizes = layer_sizes

        # layers is with size [[W0, b0], [W1, b1], ..., [Wn, bn]]
        self.layers = []
        # params is with size [W0, b0, W1, b1, ..., Wn, bn]
        self.params = []

        for i in xrange(len(layer_sizes) + 1):
            if i == 0:
                self.layers.append([glorot_uniform(shape=(self.input_size, self.layer_sizes[i]),
                                                   name='W_hidden_%d' % i),
                                    init_bias(size=self.layer_sizes[i], name='b_hidden_%d' % i)])
                self.params.extend(self.layers[-1])
            elif 0 < i < len(layer_sizes):
                self.layers.append([glorot_uniform(shape=(self.layer_sizes[i - 1], self.layer_sizes[i]),
                                                   name='W_hidden_%d' % i),
                                    init_bias(size=self.layer_sizes[i], name='b_hidden_%d' % i)])
                self.params.extend(self.layers[-1])
            else:
                self.layers.append([glorot_uniform(shape=(self.layer_sizes[i - 1], self.output_size),
                                                   name='W_hidden_%d' % i),
                                    init_bias(size=self.output_size, name='b_hidden_%d' % i)])
                self.params.extend(self.layers[-1])

        # The weight matrix for memory input to the controller
        self.W_read_hidden = glorot_uniform(shape=(mem_width, layer_sizes[0]), name='W_hidden_read')
        self.params.append(self.W_read_hidden)

        self.activations = []
        for i in range(len(layer_sizes)):
            if i == 0:
                # activations[0] with size (batch, layer_size[0])
                self.activations.append(T.tanh(T.dot(self.X, self.layers[i][0]) +
                                               T.dot(self.read_input, self.W_read_hidden) +
                                               self.layers[i][1]))
            else:  # i = 1, 2, ..., len(layer_sizes) - 1
                self.activations.append(T.tanh(T.dot(self.activations[-1],
                                                     self.layers[i][0]) +
                                               self.layers[i][1]))
        # Activations of the last hidden layer
        # with size (batch, layer_sizes[-1])
        self.fin_hidden = self.activations[-1]
        # Outputs of the controller
        self.output = T.nnet.sigmoid(T.dot(self.fin_hidden,
                                           self.layers[-1][0]) +
                                     self.layers[-1][1])





