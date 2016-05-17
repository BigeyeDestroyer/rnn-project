import theano
import theano.tensor as T
import numpy
from common.utils import *


class ReadHead(object):
    def __init__(self, number=0, input_size=100, mem_size=128,
                 mem_width=20, shift_width=3):
        """
        :type number: int
        :param number: the layer number

        :type input_size: int
        :param input_size: usually the hidden layer from controller

        :type mem_size: int
        :param mem_size: rows of the memory matrix

        :type mem_width: int
        :param mem_width: length of a single row in the memory matrix

        :type shift_width: int
        :param shift_width: width to perform shift operation
        """
        self.number = number
        self.input_size = input_size
        self.mem_size = mem_size
        self.mem_width = mem_width
        self.shift_width = shift_width

        # Initialize the params
        # 1. Content Addressing
        self.W_key = glorot_uniform(shape=(self.input_size,
                                           self.mem_width),
                                    name='W_key_read%d' % number)
        self.b_key = init_bias(size=mem_width,
                               name='b_key_read%d' % number)
        self.W_beta = init_bias(size=input_size,
                                name='W_beta_read%d' % number)
        self.b_beta = init_scalar(value=0.0,
                                  name='b_beta_read%d' % number)

        # 2. Interpolation
        self.W_g = init_bias(size=input_size,
                             name='W_g_read%d' % number)
        self.b_g = init_scalar(value=0.0,
                               name='b_g_read%d' % number)

        # 3. Convolutional Shift
        self.W_shift = glorot_uniform(shape=(self.input_size,
                                             self.shift_width),
                                      name='W_shift_read%d' % number)
        self.b_shift = init_bias(size=shift_width,
                                 name='b_shift_read%d' % number)

        # 4. Sharpening
        self.W_gamma = init_bias(size=input_size,
                                 name='W_gamma_read%d' % number)
        self.b_gamma = init_scalar(value=0.0,
                                   name='b_gamma_read%d' % number)

        self.params = [self.W_key, self.b_key, self.W_beta, self.b_beta,
                       self.W_g, self.b_g,
                       self.W_shift, self.b_shift,
                       self.W_gamma, self.b_gamma]

    def step_readhead(self, X):
        """
        :type X: tensor variable with size (batch, input_size) at time t
        :param X: input to head, usually from last layer of controller
                  e.g., if the controller is LSTM, then input_size is
                  the hidden size of the LSTM's top layer
        """
        # Calculate the Head's outputs
        """ 1. Content Addressing: key_t and beta_t

            key_t  : (batch, mem_width)
            beta_t : (batch, )
        """
        key_t = T.dot(X, self.W_key) + self.b_key
        beta_t = T.nnet.softplus(T.dot(X, self.W_beta) + self.b_beta)

        """ 2. Interpolation: g_t

            g_t : (batch, )
        """
        g_t = T.nnet.sigmoid(T.dot(X, self.W_g) + self.b_g)

        """ 3. Convolutional Shift

            shift_t : (batch, shift_width)
        """
        shift_t = T.nnet.softmax(T.dot(X, self.W_shift) +
                                 self.b_shift)

        """ 4. Sharpening, >= 1
            gamma_t : (batch, )
        """
        gamma_t = T.nnet.softplus(T.dot(X, self.W_gamma) +
                                  self.b_gamma)

        # Collect all the outputs of head
        return key_t, beta_t, g_t, shift_t, gamma_t


class WriteHead(object):
    def __init__(self, number=0, input_size=100, mem_size=128,
                 mem_width=20, shift_width=3):
        """
        :type number: int
        :param number: the layer number

        :type input_size: int
        :param input_size: usually the hidden layer from controller

        :type mem_size: int
        :param mem_size: rows of the memory matrix

        :type mem_width: int
        :param mem_width: length of a single row in the memory matrix

        :type shift_width: int
        :param shift_width: width to perform shift operation
        """
        self.number = number
        self.input_size = input_size
        self.mem_size = mem_size
        self.mem_width = mem_width
        self.shift_width = shift_width

        # Initialize the params
        # 1. Content Addressing
        self.W_key = glorot_uniform(shape=(self.input_size,
                                           self.mem_width),
                                    name='W_key_write%d' % number)
        self.b_key = init_bias(size=mem_width,
                               name='b_key_write%d' % number)
        self.W_beta = init_bias(size=input_size,
                                name='W_beta_write%d' % number)
        self.b_beta = init_scalar(value=0.0,
                                  name='b_beta_write%d' % number)

        # 2. Interpolation
        self.W_g = init_bias(size=input_size,
                             name='W_g_write%d' % number)
        self.b_g = init_scalar(value=0.0,
                               name='b_g_write%d' % number)

        # 3. Convolutional Shift
        self.W_shift = glorot_uniform(shape=(self.input_size,
                                             self.shift_width),
                                      name='W_shift_write%d' % number)
        self.b_shift = init_bias(size=shift_width,
                                 name='b_shift_write%d' % number)

        # 4. Sharpening
        self.W_gamma = init_bias(size=input_size,
                                 name='W_gamma_write%d' % number)
        self.b_gamma = init_scalar(value=0.0,
                                   name='b_gamma_write%d' % number)

        # 5. Erase and Add vector
        self.W_erase = init_weights(shape=(input_size, mem_width),
                                    name='W_erase_write%d' % number)
        self.b_erase = init_bias(size=mem_width,
                                 name='b_erase_write%d' % number)
        self.W_add = init_weights(shape=(input_size, mem_width),
                                  name='W_add_write%d' % number)
        self.b_add = init_bias(size=mem_width,
                               name='b_add_write%d' % number)

        self.params = [self.W_key, self.b_key, self.W_beta, self.b_beta,
                       self.W_g, self.b_g,
                       self.W_shift, self.b_shift,
                       self.W_gamma, self.b_gamma,
                       self.W_erase, self.b_erase, self.W_add, self.b_add]

    def step_writehead(self, X):
        """
        :type X: tensor variable with size (batch, input_size)
        :param X: the input to head, usually from last layer of controller
        """

        # Calculate the Head's outputs
        """ 1. Content Addressing: key_t and beta_t

            key_t  : (batch, mem_width)
            beta_t : (batch, )
        """
        key_t = T.dot(X, self.W_key) + self.b_key
        beta_t = T.nnet.softplus(T.dot(X, self.W_beta) + self.b_beta)

        """ 2. Interpolation: g_t

            g_t : (batch, )
        """
        g_t = T.nnet.sigmoid(T.dot(X, self.W_g) + self.b_g)

        """ 3. Convolutional Shift

            shift_t : (batch, shift_width)
        """
        shift_t = T.nnet.softmax(T.dot(X, self.W_shift) +
                                 self.b_shift)

        """ 4. Sharpening, >= 1
            gamma_t : (batch, )
        """
        gamma_t = T.nnet.softplus(T.dot(X, self.W_gamma) +
                                  self.b_gamma)

        """ 5. Erase and Add vector
            erase_t : (batch, mem_width)
            add_t   : (batch, mem_width)
        """
        erase_t = T.nnet.sigmoid(T.dot(X, self.W_erase) +
                                 self.b_erase)
        add_t = T.dot(X, self.W_add) + self.b_add

        # Collect all the outputs of head
        return key_t, beta_t, g_t, shift_t, gamma_t, erase_t, add_t
