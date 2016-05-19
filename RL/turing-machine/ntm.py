from memory import *
import numpy

# params for heads
# number is left for the initialization
controller_size = 100
mem_size = 128
mem_width = 20
shift_width = 3

# params for controller
input_dim = 8
output_dim = 8
layer_sizes = [100]

# params for memory
similarity = cosine_sim
num_reads = 1
num_writes = 1
batch_size = 5

# initial the params
params = []
controller = ControllerLSTM(input_size=input_dim, output_size=output_dim, mem_size=mem_size,
                            mem_width=mem_width, layer_sizes=layer_sizes, num_heads=num_reads)
memory = Memory(batch_size=batch_size, num_read_heads=1,
                num_write_heads=1, layer_sizes=[100])
W_output = init_weights(shape=(layer_sizes[-1], output_dim), name='W_output')
b_output = init_bias(size=output_dim, name='b_output')

params.extend(controller.params)
params.extend(memory.params)
params.extend([W_output, b_output])


# Below are the body of the step function
x_t = T.matrix('x_tm1')  # current input, (batch, input_size)

""" state contains: M_tm1, w_read_tm1_list, w_write_tm1_list
                    read_tm1_list, c_tm1_list, h_tm1_list
"""
M_tm1 = T.tensor3('M_tm1')  # previous memory, (mem_size, mem_width)
# previous read weight, a list of (batch, mem_size) array
w_read_tm1_list = [T.matrix('w_read_tm1' + str(h)) for h in range(num_reads)]
# previous write weight, a list of (batch, mem_size) array
w_write_tm1_list = [T.matrix('w_write_tm1' + str(h)) for h in range(num_writes)]
read_tm1_list = [T.matrix('read_tm1' + str(h)) for h in range(num_reads)]
c_tm1_list = [T.matrix('c_tm1_%d' % l) for l in xrange(len(layer_sizes))]
h_tm1_list = [T.matrix('h_tm1_%d' % l) for l in xrange(len(layer_sizes))]

c_t_list, h_t_list = controller.step(x_t, read_tm1_list,
                                     c_tm1_list, h_tm1_list)

last_hidden = h_t_list[-1]  # with size (batch_size, layer_sizes[-1])

M_t, w_read_t_list, w_write_t_list, read_t_list = \
    memory.step(M_tm1, w_read_tm1_list, w_write_tm1_list, last_hidden)

# test
inputs = [x_t] + [M_tm1] + w_read_tm1_list + w_write_tm1_list + \
         read_tm1_list + c_tm1_list + h_tm1_list

outputs = [M_t] + w_read_t_list + w_write_t_list + \
          read_t_list + c_t_list + h_t_list

fn_test = theano.function(inputs=inputs,
                          outputs=outputs)

x_data = numpy.random.randn(batch_size, input_dim)
M_data = numpy.random.randn(batch_size, mem_size, mem_width)
w_read_data = numpy.random.randn(batch_size, mem_size)
w_write_data = numpy.random.randn(batch_size, mem_size)
read_data = numpy.random.randn(batch_size, mem_width)
c_data = numpy.random.randn(batch_size, layer_sizes[-1])
h_data = numpy.random.randn(batch_size, layer_sizes[-1])

M_out, w_read_out, w_write_out, read_out, c_out, h_out = \
    fn_test(x_data, M_data, w_read_data, w_write_data, read_data, c_data, h_data)

print M_out.shape
print w_read_out.shape
print w_write_out.shape
print read_out.shape
print c_out.shape
print h_out.shape






















