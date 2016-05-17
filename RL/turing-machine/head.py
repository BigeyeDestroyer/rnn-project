import theano
import theano.tensor as T
import numpy
from common.utils import *
import scipy

def cosine_sim(k, M):
    """
    :type k: tensor variable with size (batch_size, mem_width)
    :param k: input to calculate similarity

    :type M: tensor variable with size (mem_size, mem_width)
    :param M: memory matrix

    :return: similarity measure with size (batch_size, mem_size)
    """
    k_lengths = T.sqrt(T.sum(k ** 2, axis=1)).dimshuffle((0, 'x'))
    k_unit = k / (k_lengths + 1e-5)

    M_lengths = T.sqrt(T.sum(M ** 2, axis=1)).dimshuffle((0, 'x'))
    M_unit = M / (M_lengths + 1e-5)
    return T.dot(k_unit, T.transpose(M_unit))


# In this version, read heads and write heads are independent
class ReadHead(object):
    def __init__(self, number=0, input_size=100, mem_size=128,
                 mem_width=20, shift_width=3, similarity=cosine_sim):
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

        :type similarity: function handle
        :param similarity: function to perform similarity metric
        """
        self.number = number
        self.input_size = input_size
        self.mem_size = mem_size
        self.mem_width = mem_width
        self.shift_width = shift_width
        self.similarity = similarity

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

    def step(self, x_t, w_tm1, M_t):
        """
        :type x_t: tensor variable with size (batch, input_size) at time t
        :param x_t: input to head, usually from last layer of controller
                    e.g., if the controller is LSTM, then input_size is
                    the hidden size of the LSTM's top layer

        :type w_tm1: tensor variable with size (batch_size, mem_size)
        :param w_tm1: head's weight at last time step

        :type M_t: tensor variable with size (mem_size, mem_width) at time t
        :param M_t: memory matrix at time t
        """
        # Calculate the Head's outputs
        """ 1. Content Addressing: key_t and beta_t

            key_t  : (batch, mem_width)
            beta_t : (batch, )
        """
        key_t = T.dot(x_t, self.W_key) + self.b_key
        beta_t = T.nnet.softplus(T.dot(x_t, self.W_beta) + self.b_beta)

        """ 2. Interpolation: g_t

            g_t : (batch, )
        """
        g_t = T.nnet.sigmoid(T.dot(x_t, self.W_g) + self.b_g)

        """ 3. Convolutional Shift

            shift_t : (batch, shift_width)
        """
        shift_t = T.nnet.softmax(T.dot(x_t, self.W_shift) +
                                 self.b_shift)

        """ 4. Sharpening, >= 1
            gamma_t : (batch, )
        """
        gamma_t = T.nnet.softplus(T.dot(x_t, self.W_gamma) +
                                  self.b_gamma)

        # Calculates the final output : w_t
        """ 1. Content Addressing: w_c_t, with size (batch, mem_size)
        """
        # After similarity, the size is (batch, mem_size)
        # beta_t is with size (batch, ),
        #        dimshuffle makes it with size (batch, 1)
        # Thus, the multi will broadcast along dim1 of similarity
        w_c_t = T.nnet.softmax(beta_t.dimshuffle(0, 'x') *
                               self.similarity(key_t, M_t))

        """ 2. Interpolation: w_g_t, with size (batch, mem_size)
        """
        # g_t with size (batch, )
        # w_c_t with size (batch, mem_size)
        # w_tm1 with size (batch, mem_size)
        w_g_t = g_t.dimshuffle(0, 'x') * w_c_t + \
                (1 - g_t.dimshuffle(0, 'x')) * w_tm1

        """ 3. Convolutional Shift: w_tiled_t with size (batch, mem_size)

            shift_conv : with size (shift_width, mem_size)
                         each row represents a shift, one of +1, 0, -1
                         suppose shift_width = 3, then the operations'
                         order of the rows of shift_conv are +1, 0, -1

                         scipy.linalg.circulant receives a vector with size (N, )
                         and returns a matrix with size (N, N)
                         each column is generated by applying the +1 operation on
                         the previous column, and the first column is the input

        """
        # convolutions applied to the 'w' generated by heads
        # shift_conv with size (shift_width, mem_size), from
        # top to bottom with operations +1, 0, -1
        shift_conv = scipy.linalg.circulant(numpy.arange(self.mem_size)).T[
                         numpy.arange(-(self.shift_width // 2),
                                      (self.shift_width // 2) + 1)][::-1]

        # shift_t with size (batch_size, shift_width)
        #  after T.tile, shift is with size (batch_size, shift_width, mem_size)
        #  w_g_t[shift_conv] with size (batch_size, shift_width, mem_size)
        # w_tiled_t is with size (batch, mem_size)
        w_tiled_t = T.sum(T.tile(T.reshape(shift_t, (shift_t.shape[0], shift_t.shape[1], 1)),
                                 (1, 1, self.mem_size)) * w_g_t[:, shift_conv], axis=1)

        """ 4. Sharpening: w_t with size (batch, mem_size)
        """
        # w_tiled_t with size (batch, mem_size)
        # gamma_t with size (batch, )
        w_sharp = w_tiled_t ** T.tile(T.reshape(gamma_t, (gamma_t.shape[0], 1)),
                                      (1, self.mem_size))
        w_t = w_sharp / T.reshape(T.sum(w_sharp, axis=1), (w_sharp.shape[0], 1))

        return w_t


class WriteHead(object):
    def __init__(self, number=0, input_size=100, mem_size=128,
                 mem_width=20, shift_width=3, similarity=cosine_sim):
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

        :type similarity: function handle
        :param similarity: function to perform similarity metric
        """
        self.number = number
        self.input_size = input_size
        self.mem_size = mem_size
        self.mem_width = mem_width
        self.shift_width = shift_width
        self.similarity = similarity

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

    def step(self, x_t, w_tm1, M_t):
        """
        :type x_t: tensor variable with size (batch, input_size) at time t
        :param x_t: input to head, usually from last layer of controller
                    e.g., if the controller is LSTM, then input_size is
                    the hidden size of the LSTM's top layer

        :type w_tm1: tensor variable with size (batch_size, mem_size)
        :param w_tm1: head's weight at last time step

        :type M_t: tensor variable with size (mem_size, mem_width) at time t
        :param M_t: memory matrix at time t
        """

        # Calculate the Head's outputs
        """ 1. Content Addressing: key_t and beta_t

            key_t  : (batch, mem_width)
            beta_t : (batch, )
        """
        key_t = T.dot(x_t, self.W_key) + self.b_key
        beta_t = T.nnet.softplus(T.dot(x_t, self.W_beta) + self.b_beta)

        """ 2. Interpolation: g_t

            g_t : (batch, )
        """
        g_t = T.nnet.sigmoid(T.dot(x_t, self.W_g) + self.b_g)

        """ 3. Convolutional Shift

            shift_t : (batch, shift_width)
        """
        shift_t = T.nnet.softmax(T.dot(x_t, self.W_shift) +
                                 self.b_shift)

        """ 4. Sharpening, >= 1
            gamma_t : (batch, )
        """
        gamma_t = T.nnet.softplus(T.dot(x_t, self.W_gamma) +
                                  self.b_gamma)

        """ 5. Erase and Add vector
            erase_t : (batch, mem_width)
            add_t   : (batch, mem_width)
        """
        erase_t = T.nnet.sigmoid(T.dot(x_t, self.W_erase) +
                                 self.b_erase)
        add_t = T.dot(x_t, self.W_add) + self.b_add

        # Calculates the final output : w_t
        """ 1. Content Addressing: w_c_t, with size (batch, mem_size)
        """
        # After similarity, the size is (batch, mem_size)
        # beta_t is with size (batch, ),
        #        dimshuffle makes it with size (batch, 1)
        # Thus, the multi will broadcast along dim1 of similarity
        w_c_t = T.nnet.softmax(beta_t.dimshuffle(0, 'x') *
                               self.similarity(key_t, M_t))

        """ 2. Interpolation: w_g_t, with size (batch, mem_size)
        """
        # g_t with size (batch, )
        # w_c_t with size (batch, mem_size)
        # w_tm1 with size (batch, mem_size)
        w_g_t = g_t.dimshuffle(0, 'x') * w_c_t + \
                (1 - g_t.dimshuffle(0, 'x')) * w_tm1

        """ 3. Convolutional Shift: w_tiled_t with size (batch, mem_size)

            shift_conv : with size (shift_width, mem_size)
                         each row represents a shift, one of +1, 0, -1
                         suppose shift_width = 3, then the operations'
                         order of the rows of shift_conv are +1, 0, -1

                         scipy.linalg.circulant receives a vector with size (N, )
                         and returns a matrix with size (N, N)
                         each column is generated by applying the +1 operation on
                         the previous column, and the first column is the input

        """
        # convolutions applied to the 'w' generated by heads
        # shift_conv with size (shift_width, mem_size), from
        # top to bottom with operations +1, 0, -1
        shift_conv = scipy.linalg.circulant(numpy.arange(self.mem_size)).T[
                         numpy.arange(-(self.shift_width // 2),
                                      (self.shift_width // 2) + 1)][::-1]

        # shift_t with size (batch_size, shift_width)
        #  after T.tile, shift is with size (batch_size, shift_width, mem_size)
        #  w_g_t[shift_conv] with size (batch_size, shift_width, mem_size)
        # w_tiled_t is with size (batch, mem_size)
        w_tiled_t = T.sum(T.tile(T.reshape(shift_t, (shift_t.shape[0], shift_t.shape[1], 1)),
                                 (1, 1, self.mem_size)) * w_g_t[:, shift_conv], axis=1)

        """ 4. Sharpening: w_t with size (batch, mem_size)
        """
        # w_tiled_t with size (batch, mem_size)
        # gamma_t with size (batch, )
        w_sharp = w_tiled_t ** T.tile(T.reshape(gamma_t, (gamma_t.shape[0], 1)),
                                      (1, self.mem_size))
        w_t = w_sharp / T.reshape(T.sum(w_sharp, axis=1), (w_sharp.shape[0], 1))

        # Collect all the outputs of head
        return w_t, erase_t, add_t
