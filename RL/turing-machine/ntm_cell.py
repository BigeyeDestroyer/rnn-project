from memory import *


class NTMCell(object):
    def __init__(self, input_dim=8, output_dim=8, mem_size=128,
                 mem_width=20, layer_sizes=[100], num_reads=1,
                 batch_size=1, num_writes=1, shift_width=3, eps=1e-12):

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
        self.eps = eps

        self.depth = 0  # to store the states along time steps
        self.states = []

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

    def step(self, x_t):
        """
        :type x_t: tensor variable with size (batch_size, input_dim)
        :param x_t: input at time step t
        """

        if self.depth == 0:  # which means we need initialization
            state = self.initial_state()
            self.depth += 1
            self.states.append(state)
        else:
            state = self.states[-1]

        M_tm1 = state['M']
        w_read_tm1_list = state['w_read']
        w_write_tm1_list = state['w_write']
        read_tm1_list = state['read']
        c_tm1_list = state['c']
        h_tm1_list = state['h']

        c_t_list, h_t_list = self.controller.step(x_t, read_tm1_list,
                                                  c_tm1_list, h_tm1_list)

        last_hidden = h_t_list[-1]  # with size (batch_size, layer_sizes[-1])

        M_t, w_read_t_list, w_write_t_list, read_t_list = \
            self.memory.step(M_tm1, w_read_tm1_list, w_write_tm1_list, last_hidden)

        # Output to the environment
        # with size (batch_size, output_dim)
        output_t = T.nnet.sigmoid(T.dot(last_hidden, self.W_output) +
                                  self.b_output)

        state = {
            'M': M_t,
            'w_read': w_read_t_list,
            'w_write': w_write_t_list,
            'read': read_t_list,
            'c': c_t_list,
            'h': h_t_list
        }

        self.depth += 1
        self.states.append(state)

        return output_t

    def initial_state(self, dummy_value=0.0):
        # Initial memory, with size (mem_size, mem_width)
        # don't forget cast dummy to floatX
        M_init = T.alloc(floatX(dummy_value), self.batch_size,
                         self.mem_size, self.mem_width)

        w_read_init_list = []
        read_init_list = []
        for idx in xrange(self.num_reads):
            # Initial read weight, with size (batch_size, mem_size)
            w_read_init = T.nnet.softmax(T.alloc(floatX(self.eps),
                                                 self.batch_size, self.mem_size))
            # Initial read memory, with size (batch_size, mem_width)
            read_init = T.tanh(T.alloc(floatX(dummy_value),
                                       self.batch_size, self.mem_width))

            w_read_init_list.append(w_read_init)
            read_init_list.append(read_init)

        w_write_init_list = []
        for idx in xrange(self.num_writes):
            # Initial write weight, with size (batch_size, mem_size)
            w_write_init = T.nnet.softmax(T.alloc(floatX(self.eps),
                                                  self.batch_size, self.mem_size))
            w_write_init_list.append(w_write_init)

        # cell and hidden layers of the LSTM controller
        c_init_list = []
        h_init_list = []
        for idx in xrange(len(self.layer_sizes)):
            c_init = T.tanh(T.alloc(floatX(dummy_value),
                                    self.batch_size, self.layer_sizes[idx]))
            h_init = T.tanh(T.alloc(floatX(dummy_value),
                                    self.batch_size, self.layer_sizes[idx]))
            c_init_list.append(c_init)
            h_init_list.append(h_init)

        state = {
            'M': M_init,
            'w_read': w_read_init_list,
            'w_write': w_write_init_list,
            'read': read_init_list,
            'c': c_init_list,
            'h': h_init_list
        }
        return state

    # M: D tensor with size (batch_size, mem_size, mem_width)
    def get_memory(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['M']

    # w_read: a list of 2D tensors, each with size (batch_size, mem_size)
    def get_read_weights(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['w_read']

    # w_write: a list of 2D tensors, each with size (batch_size, mem_size)
    def get_write_weights(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['w_write']

    # read: a list of 2D tensors, each with size (batch_size, mem_width)
    def get_read_vectors(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['read']



























