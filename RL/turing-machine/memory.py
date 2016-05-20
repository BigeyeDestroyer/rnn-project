from head import *
from controller import *


class Memory(object):
    def __init__(self, batch_size, num_read_heads=1, num_write_heads=1,
                 layer_sizes=[100], mem_size=128, mem_width=20, shift_width=3):
        """
        :type batch_size: int
        :param batch_size: for iteration in the 'theano tensor' environment

        :type num_read_heads: int
        :param num_read_heads:

        :type num_write_heads: int
        :param num_write_heads:

        :type layer_sizes: list of ints
        :param layer_sizes: indicating the hidden
                            dimensions of the controller
        """
        self.batch_size = batch_size
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.layer_sizes = layer_sizes
        self.mem_size = mem_size
        self.mem_width = mem_width
        self.shift_width = shift_width
        self.params = []

        # 1. Build all the read heads and write heads
        self.read_heads = []  # read_heads, list of read heads
        for idx in xrange(self.num_read_heads):
            self.read_heads.append(Head(idx=idx, last_dim=self.layer_sizes[-1], mem_size=self.mem_size,
                                        mem_width=self.mem_width, shift_width=self.shift_width, is_write=False))

            self.params.extend(self.read_heads[-1].params)

        self.write_heads = []  # write_heads, list of write heads
        for idx in xrange(self.num_write_heads):
            self.write_heads.append(Head(idx=idx, last_dim=self.layer_sizes[-1], mem_size=self.mem_size,
                                         mem_width=self.mem_width, shift_width=self.shift_width, is_write=True))
            self.params.extend(self.write_heads[-1].params)

    def step(self, M_tm1, w_read_tm1_list, w_write_tm1_list, last_hidden):
        """
        :type M_tm1: tensor variable with size (batch_size, mem_size, mem_width)
        :param M_tm1: memory state from last time step

        :type w_read_tm1_list: list of tensor variables, each
                               with size (batch_size, mem_size)
        :param w_read_tm1_list: weights of read heads at last time step

        :type w_write_tm1_list: list of tensor variables, each
                                with size (batch_size, mem_size)
        :param w_write_tm1_list: weights of write heads at last time step

        :type last_hidden: tensor variable with size (batch_size, last_dim)
        :param last_hidden: last layer's outputs from the comtroller at
                            current time step
        :return:
        :type M_t: tensor variable with size (batch_size, mem_size, mem_width)
        :param M_t: memory state at current time step

        :type w_read_t_list: list of tensor variables, each
                             with size (batch_size, mem_size)
        :param w_read_t_list: weights of read heads at current time step

        :type w_write_t_list: list of tensor variables, each
                              with size (batch_size, mem_size)
        :param w_write_t_list: weights of write heads at current time step

        :type read_t_list: list of tensor variables, each
                           with size (batch_size, mem_width)
        :param read_t_lsit: memory read at time step t
        """
        w_read_t_list = []
        read_t_list = []
        # Get the read heads' output
        for idx in xrange(self.num_read_heads):
            w_read_t, read_t = self.read_heads[idx].step(M_tm1=M_tm1,
                                                         w_tm1=w_read_tm1_list[idx],
                                                         last_hidden=last_hidden,
                                                         batch_size=self.batch_size)
            w_read_t_list.append(w_read_t)
            read_t_list.append(read_t)

        w_write_t_list = []
        erase_t_list = []
        add_t_list = []

        def batch_nulti(w, W, batch_size):
            """
            :type w: tensor variable, with size (batch_size, mem_size)
            :param w: on the left side of outer-product

            :type W: tensor variable, with size (batch_size, mem_width)
            :param W: on the right side of outer-product

            :type batch_size: int
            :param batch_size: for iteration under T environment

            :return:
            :type : tensor variable with size (batch_size, mem_size, mem_width)
            """
            multi_list = []
            for idx in xrange(batch_size):
                multi_list.append(T.dot(w[idx, :].dimshuffle(0, 'x'),
                                        W[idx, :].dimshuffle('x', 0)).dimshuffle('x', 0, 1))
            return T.concatenate(multi_list, axis=0)

        # M_t_erases and M_t_adds are lists of tensor3
        M_t_erases = []
        M_t_adds = []
        # Get the write heads' output
        for idx in xrange(self.num_write_heads):
            # w_write_t with size (bath_size, mem_size)
            # erase_t with size (batch_size, mem_width)
            # add_t with size (batch_size, mem_width)
            w_write_t, erase_t, add_t = self.write_heads[idx].step(M_tm1=M_tm1,
                                                                   w_tm1=w_write_tm1_list[idx],
                                                                   last_hidden=last_hidden,
                                                                   batch_size=self.batch_size)
            w_write_t_list.append(w_write_t)
            erase_t_list.append(erase_t)
            add_t_list.append(add_t)

            M_t_erases.append(1. - batch_nulti(w=w_write_t, W=erase_t,
                                               batch_size=self.batch_size))
            M_t_adds.append(batch_nulti(w=w_write_t, W=add_t,
                                        batch_size=self.batch_size))
        # M_t_erase is with size (batch_size, mem_size, mem_width)
        M_t_erase = reduce(lambda x, y: x*y, M_t_erases)
        # M_t_add is with size (batch_size, mem_size, mem_width)
        M_t_add = reduce(lambda x, y: x + y, M_t_adds)

        M_t = M_tm1 * M_t_erase + M_t_add

        return M_t, w_read_t_list, w_write_t_list, read_t_list