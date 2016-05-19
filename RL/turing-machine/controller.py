import theano
import theano.tensor as T
import numpy
from common.utils import *


class ControllerLSTM(object):
    def __init__(self, input_size=8, output_size=8, mem_size=128,
                 mem_width=20, layer_sizes=[100], num_heads=3):
        """
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

        :type num_heads: int
        :param num_heads: number of read heads
        """
        self.input_size = input_size
        self.output_size = output_size
        self.mem_size = mem_size
        self.mem_width = mem_width
        self.layer_sizes = layer_sizes
        self.num_heads = num_heads

        # The read head related weights
        self.W_read_list = [init_weights(shape=(self.mem_width, 4 * self.layer_sizes[0]),
                                         name='W_read_%d' % h) for h in range(self.num_heads)]

        # layers is with ndarrays like [[W0, U0, b0], [W1, U1, b1], ..., [Wn, Un, bn]]
        self.layers = []
        # params is with ndarrays like [W0, U0, b0, W1, U1, b1, ..., Wn, Un, bn]
        self.params = []
        for layer_idx in xrange(len(layer_sizes)):
            if layer_idx == 0:
                W = init_weights(shape=(self.input_size, 4 * self.layer_sizes[layer_idx]),
                                 name='W_lstm_%d' % layer_idx)
                U = theano.shared(value=numpy.concatenate((ortho_weight(ndim=self.layer_sizes[layer_idx]),
                                                           ortho_weight(ndim=self.layer_sizes[layer_idx]),
                                                           ortho_weight(ndim=self.layer_sizes[layer_idx]),
                                                           ortho_weight(ndim=self.layer_sizes[layer_idx])),
                                                          axis=1),
                                  name='U_lstm_%d' % layer_idx)
                b = init_bias(size=4 * self.layer_sizes[layer_idx], name='b_lstm_%d' % layer_idx)
            else:
                W = init_weights(shape=(self.layer_sizes[layer_idx - 1], 4 * self.layer_sizes[layer_idx]),
                                 name='W_lstm_%d' % layer_idx)

                U = theano.shared(value=numpy.concatenate((ortho_weight(ndim=self.layer_sizes[layer_idx]),
                                                           ortho_weight(ndim=self.layer_sizes[layer_idx]),
                                                           ortho_weight(ndim=self.layer_sizes[layer_idx]),
                                                           ortho_weight(ndim=self.layer_sizes[layer_idx])),
                                                          axis=1),
                                  name='U_lstm_%d' % layer_idx)
                b = init_bias(size=4 * self.layer_sizes[layer_idx], name='b_lstm_%d' % layer_idx)

            self.layers.append([W, U, b])
            self.params.extend(self.layers[-1])
        # Add the read head related weights to params' list
        self.params.extend(self.W_read_list)

    def step(self, x_t, read_tm1_list, c_tm1_list, h_tm1_list):
        """
        Carry one step computation for the controller

        Parameters
        ----------
        :type x_t: generated as x_t = T.matrix('x_t') , tensor variable
        :param x_t: input at time t, with size (batch, input_size)

        :type read_tm1_list: [T.matrix('read_tm1_%d' % h) for h in xrange(num_heads)]
        :param read_tm1_list: a list of tensor variables, each with size (batch, mem_width)

        :type c_tm1_list: [T.matrix('c_tm1_%d' % l) for l in xrange(len(layer_sizes))]
        :param c_tm1_list: a list of tensor variables, each with size (batch, layer_sizes[l])

        :type h_tm1_list: [T.matrix('h_tm1_%d' % l) for l in xrange(len(layer_sizes))]
        :param h_tm1_list: a list of tensor variables, each with size (batch, layer_sizes[l])

        :returns
        :type c_t_list: a list of tensor variables each with size (batch, layer_sizes[l])
        :param c_t_list: value of cell in LSTM at time t

        :type h_t_list: a list of tensor variables each with size (batch, layer_sizes[l])
        :param: h_t_list: value of the hidden states of LSTM at time t
        """
        def _slice(x, n, dim):
            if x.ndim == 3:
                return x[:, :, n * dim: (n + 1) * dim]
            return x[:, n * dim: (n + 1) * dim]

        # c_t_list and h_t_list are calculated
        c_t_list = []  # each LSTM layer's c
        h_t_list = []  # each LSTM layer's h

        for layer_idx in xrange(len(self.layer_sizes)):
            W, U, b = self.layers[layer_idx]  # extract params from the list: layers
            if layer_idx == 0:
                # preact with size (batch, 4 * layer_sizes[layer_idx])
                preact = T.dot(x_t, W) + T.dot(h_tm1_list[layer_idx], U) + b
                # if layer0, then we should also consider inputs from read head
                for h in xrange(self.num_heads):
                    preact += T.dot(read_tm1_list[h], self.W_read_list[h])
            else:
                # preact with size (batch, 4 * layer_sizes[layer_idx])
                preact = T.dot(h_t_list[-1], W) + T.dot(h_tm1_list[layer_idx], U) + b

            i = T.nnet.sigmoid(_slice(preact, 0, self.layer_sizes[layer_idx]))
            f = T.nnet.sigmoid(_slice(preact, 1, self.layer_sizes[layer_idx]))
            o = T.nnet.sigmoid(_slice(preact, 2, self.layer_sizes[layer_idx]))
            c_tilde = T.tanh(_slice(preact, 3, self.layer_sizes[layer_idx]))

            c_t = f * c_tm1_list[layer_idx] + i * c_tilde
            h_t = o * T.tanh(c_t)

            c_t_list.append(c_t)
            h_t_list.append(h_t)

        return c_t_list, h_t_list

