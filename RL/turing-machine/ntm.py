from memory import *


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


# since X, M, W_read and W_write are changing through time
# we firstly try to write their step-wise function
x_t = T.matrix('x_tm1')  # current input, (batch, input_size)

M_tm1 = T.tensor3('M_tm1')  # previous memory, (mem_size, mem_width)
# previous read weight, a list of (batch, mem_size) array
w_read_tm1_list = [T.matrix('w_read_tm1' + str(h)) for h in range(num_reads)]
# previous write weight, a list of (batch, mem_size) array
w_write_tm1_list = [T.matrix('w_write_tm1' + str(h)) for h in range(num_writes)]
read_tm1_list = [T.matrix('read_tm1' + str(h)) for h in range(num_reads)]
c_tm1_list = [T.matrix('c_tm1_%d' % l) for l in xrange(len(layer_sizes))]
h_tm1_list = [T.matrix('h_tm1_%d' % l) for l in xrange(len(layer_sizes))]




















