from memory import *
import numpy


class NTMCell(object):
    def __init__(self, input_dim=8, output_dim=8, mem_size=128,
                 mem_width=20, layer_sizes=[100], num_reads=1,
                 batch_size=5, num_writes=1, shift_width=3):

        # params for controller
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mem_size = mem_size
        self.mem_width = mem_width
        self.layer_sizes = layer_sizes
        self.num_reads = num_reads

        # params for memory
        self.batch_size = batch_size
        self.num_writes = num_writes
        self.shift_width = shift_width

        # initial the params
        self.params = []
        self.controller = ControllerLSTM(input_size=self.input_dim, output_size=self.output_dim,
                                         mem_size=self.mem_size, mem_width=self.mem_width,
                                         layer_sizes=self.layer_sizes, num_heads=self.num_reads)
        self.memory = Memory(batch_size=self.batch_size, num_read_heads=self.num_reads,
                             num_write_heads=self.num_writes, layer_sizes=self.layer_sizes)
        self.W_output = init_weights(shape=(self.layer_sizes[-1], self.output_dim), name='W_output')
        self.b_output = init_bias(size=self.output_dim, name='b_output')

        self.params.extend(self.controller.params)
        self.params.extend(self.memory.params)
        self.params.extend([self.W_output, self.b_output])

    def step(self, x_t, M_tm1, w_read_tm1_list, w_write_tm1_list,
             read_tm1_list, c_tm1_list, h_tm1_list):

        c_t_list, h_t_list = self.controller.step(x_t, read_tm1_list,
                                                  c_tm1_list, h_tm1_list)

        last_hidden = h_t_list[-1]  # with size (batch_size, layer_sizes[-1])

        M_t, w_read_t_list, w_write_t_list, read_t_list = \
            self.memory.step(M_tm1, w_read_tm1_list, w_write_tm1_list, last_hidden)
        return M_t, w_read_t_list, w_write_t_list, read_t_list, c_t_list, h_t_list























