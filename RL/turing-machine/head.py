import theano.tensor as T
from common.utils import *
import scipy


def cosine_sim(k, M, batch_size):
    """
    :type k: tensor variable with size (batch_size, mem_width)
    :param k: input to calculate similarity

    :type M: tensor variable with size (batch_size, mem_size, mem_width)
    :param M: memory matrix

    :type batch_size: int
    :param batch_size: used to iterate for each sample

    :return: similarity measure with size (batch_size, mem_size)
    """
    k_lengths = T.sqrt(T.sum(k ** 2, axis=1)).dimshuffle((0, 'x'))
    k_unit = k / (k_lengths + 1e-5)

    M_lengths = T.sqrt(T.sum(M ** 2, axis=2)).dimshuffle((0, 1, 'x'))  # with size (batch_size, mem_size, 1)
    M_unit = M / (M_lengths + 1e-5)  # with size (batch_size, mem_size, mem_width)

    list_sim = []
    for idx in range(batch_size):
        list_sim.append(T.dot(k_unit[idx, :],
                              T.transpose(M_unit[idx, :, :])).dimshuffle('x', 0))
    return T.concatenate(tensor_list=list_sim, axis=0)


# In this version, read heads and write heads are independent
class Head(object):
    def __init__(self, idx=0, last_dim=100, mem_size=128,
                 mem_width=20, shift_width=3, similarity=cosine_sim,
                 is_write=True):
        """
        :type idx: int
        :param idx: the layer number

        :type last_dim: int
        :param last_dim: dim of the controller's last layer

        :type mem_size: int
        :param mem_size: rows of the memory matrix

        :type mem_width: int
        :param mem_width: length of a single row in the memory matrix

        :type shift_width: int
        :param shift_width: width to perform shift operation

        :type similarity: function handle
        :param similarity: function to perform similarity metric

        :type is_write: bool
        :param is_write: if True, the write head; else, read head
        """
        self.last_dim = last_dim
        self.mem_size = mem_size
        self.mem_width = mem_width
        self.shift_width = shift_width
        self.similarity = similarity
        self.is_write = is_write
        if self.is_write:
            self.suffix = 'write' + str(idx)
        else:
            self.suffix = 'read' + str(idx)

        # Initialize the params
        # 1. Content Addressing
        self.W_key = glorot_uniform(shape=(self.last_dim,
                                           self.mem_width),
                                    name='W_key_' + self.suffix)
        self.b_key = init_bias(size=mem_width,
                               name='b_key_' + self.suffix)
        self.W_beta = init_bias(size=self.last_dim,
                                name='W_beta_' + self.suffix)
        self.b_beta = init_scalar(value=0.0,
                                  name='b_beta_' + self.suffix)

        # 2. Interpolation
        self.W_g = init_bias(size=self.last_dim,
                             name='W_g_' + self.suffix)
        self.b_g = init_scalar(value=0.0,
                               name='b_g_' + self.suffix)

        # 3. Convolutional Shift
        self.W_shift = glorot_uniform(shape=(self.last_dim,
                                             self.shift_width),
                                      name='W_shift_' + self.suffix)
        self.b_shift = init_bias(size=shift_width,
                                 name='b_shift_' + self.suffix)

        # 4. Sharpening
        self.W_gamma = init_bias(size=self.last_dim,
                                 name='W_gamma_' + self.suffix)
        self.b_gamma = init_scalar(value=0.0,
                                   name='b_gamma_' + self.suffix)

        self.params = [self.W_key, self.b_key, self.W_beta, self.b_beta,
                       self.W_g, self.b_g,
                       self.W_shift, self.b_shift,
                       self.W_gamma, self.b_gamma]

        if is_write:
            # 5. Erase and Add vector
            self.W_erase = init_weights(shape=(self.last_dim, self.mem_width),
                                        name='W_erase_' + self.suffix)
            self.b_erase = init_bias(size=self.mem_width,
                                     name='b_erase_' + self.suffix)
            self.W_add = init_weights(shape=(self.last_dim, self.mem_width),
                                      name='W_add_' + self.suffix)
            self.b_add = init_bias(size=self.mem_width,
                                   name='b_add_' + self.suffix)
            self.params.extend([self.W_erase, self.b_erase,
                                self.W_add, self.b_add])

    def step(self, M_tm1, w_tm1, last_hidden, batch_size):
        """
        This function computes one step of the head

        Parameters
        ----------
        :type M_tm1: tensor variable with size (batch_size, mem_size, mem_width) at time t - 1
        :param M_tm1: memory matrix at time t - 1

        :type w_tm1: tensor variable with size (batch_size, mem_size)
        :param w_tm1: head's weight at last time step

        :type last_hidden: tensor variable with size (batch, last_dim) at time t
        :param last_hidden: last hidden layer of controller, last_dim is the
                            hidden size of the LSTM's top layer

        :type batch_size: int
        :param batch_size: used in the similarity computation

        :returns

        :type w_t: tensor variable, with size (batch_size, mem_size)
        :param w_t: weights returned by heads at time t

        :type read_t; tensor variable, with size (batch_size, mem_width)
        :param read_t: only returned by read heads

        :type erase_t: tensor variable, with size (batch_size, mem_width)
        :param erase_t: only returned by write heads

        :type add_t: tensor variable, with size (batch_size, mem_width)
        :param add_t: only returned by write heads
        """
        # Calculate the Head's outputs
        """ 1. Content Addressing: key_t and beta_t

            key_t  : (batch, mem_width)
            beta_t : (batch, )
        """
        key_t = T.dot(last_hidden, self.W_key) + self.b_key
        beta_t = T.nnet.softplus(T.dot(last_hidden, self.W_beta) + self.b_beta)

        """ 2. Interpolation: g_t

            g_t : (batch, )
        """
        g_t = T.nnet.sigmoid(T.dot(last_hidden, self.W_g) + self.b_g)

        """ 3. Convolutional Shift

            shift_t : (batch, shift_width)
        """
        shift_t = T.nnet.softmax(T.dot(last_hidden, self.W_shift) +
                                 self.b_shift)

        """ 4. Sharpening, >= 1
            gamma_t : (batch, )
        """
        gamma_t = T.nnet.softplus(T.dot(last_hidden, self.W_gamma) +
                                  self.b_gamma) + 1.

        # Calculates the final output : w_t
        """ 1. Content Addressing: w_c_t, with size (batch, mem_size)
        """
        # After similarity, the size is (batch, mem_size)
        # beta_t is with size (batch, ),
        #        dimshuffle makes it with size (batch, 1)
        # Thus, the multi will broadcast along dim1 of similarity
        w_c_t = T.nnet.softmax(beta_t.dimshuffle(0, 'x') *
                               self.similarity(key_t, M_tm1, batch_size))

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

        if self.is_write:
            """ 5. Erase and Add vector
                erase_t : (batch, mem_width)
                add_t   : (batch, mem_width)
            """
            erase_t = T.nnet.sigmoid(T.dot(last_hidden, self.W_erase) +
                                     self.b_erase)
            add_t = T.dot(last_hidden, self.W_add) + self.b_add

            return w_t, erase_t, add_t
        else:
            # read_t : (batch, mem_width)
            read_batch = []
            # each multi betweem (mem_size, ) (mem_size, mem_width)
            # w_t with size (batch_size, mem_size)
            # M_tm1 with size (batch_size, mem_size, mem_width)
            for idx in xrange(batch_size):
                read_batch.append(T.dot(w_t[idx, :], M_tm1[idx, :, :]).dimshuffle('x', 0))
            read_t = T.concatenate(tensor_list=read_batch, axis=0)
            return w_t, read_t
