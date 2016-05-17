import theano
import theano.tensor as T
import numpy
from common.utils import *


class WriteHead(object):
    def __init__(self, X, number=0, input_size=100, mem_size=128,
                 mem_width=20, shift_width=3):
        """
        :type X: tensor variable with size (batch, input_size)
        :param X: the input to head, usually from last layer of controller

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
        self.X = X
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

        # Calculate the Head's outputs
        """ 1. Content Addressing: key_t and beta_t

            key_t  : (batch, mem_width)
            beta_t : (batch, )
        """
        self.key_t = T.dot(self.X, self.W_key) + self.b_key
        self.beta_t = T.nnet.softplus(T.dot(self.X, self.W_beta) + self.b_beta)

        """ 2. Interpolation: g_t

            g_t : (batch, )
        """
        self.g_t = T.nnet.sigmoid(T.dot(self.X, self.W_g) + self.b_g)

        """ 3. Convolutional Shift

            shift_t : (batch, shift_width)
        """
        self.shift_t = T.nnet.softmax(T.dot(self.X, self.W_shift) +
                                      self.b_shift)

        """ 4. Sharpening, >= 1
            gamma_t : (batch, )
        """
        self.gamma_t = T.nnet.softplus(T.dot(self.X, self.W_gamma) +
                                       self.b_gamma)

        """ 5. Erase and Add vector
            erase_t : (batch, mem_width)
            add_t   : (batch, mem_width)
        """
        self.erase_t = T.nnet.sigmoid(T.dot(self.X, self.W_erase) +
                                      self.b_erase)
        self.add_t = T.dot(self.X, self.W_add) + self.b_add

        # Collect all the outputs of head
        self.head_output = [self.key_t, self.beta_t,
                            self.g_t, self.shift_t,
                            self.gamma_t,
                            self.erase_t, self.add_t]


class ReadHead(object):
    def __init__(self, X, number=0, input_size=100, mem_size=128,
                 mem_width=20, shift_width=3):
        """
        :type X: tensor variable with size (batch, input_size)
        :param X: the input to head, usually from last layer of controller

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
        self.X = X
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

        # Calculate the Head's outputs
        """ 1. Content Addressing: key_t and beta_t

            key_t  : (batch, mem_width)
            beta_t : (batch, )
        """
        self.key_t = T.dot(self.X, self.W_key) + self.b_key
        self.beta_t = T.nnet.softplus(T.dot(self.X, self.W_beta) + self.b_beta)

        """ 2. Interpolation: g_t

            g_t : (batch, )
        """
        self.g_t = T.nnet.sigmoid(T.dot(self.X, self.W_g) + self.b_g)

        """ 3. Convolutional Shift

            shift_t : (batch, shift_width)
        """
        self.shift_t = T.nnet.softmax(T.dot(self.X, self.W_shift) +
                                      self.b_shift)

        """ 4. Sharpening, >= 1
            gamma_t : (batch, )
        """
        self.gamma_t = T.nnet.softplus(T.dot(self.X, self.W_gamma) +
                                       self.b_gamma)

        # Collect all the outputs of head
        self.head_output = [self.key_t, self.beta_t,
                            self.g_t, self.shift_t,
                            self.gamma_t]



