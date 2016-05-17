from common.utils import *
import theano
import theano.tensor as T
import numpy
from head import *
from controller import *
import scipy







# params for heads
# number is left for the initialization
controller_size = 100
mem_size = 128
mem_width = 20
shift_width = 3

# params for controller
input_size = 8
output_size = 8
layer_sizes = [100]

# params for all the read and write heads
similarity = cosine_sim
num_reads = 1
num_writes = 1

# since X, M, W_read and W_write are changing through time
# we firstly try to write their step-wise function
x_t = T.matrix('x_t')  # current input, (batch, input_size)
m_t = T.matrix('m_t')  # current memory, (mem_size, mem_width)

# current read weight, a list of (batch, mem_size) array
w_read_t = [T.matrix('w_read_t' + str(h)) for h in range(num_reads)]
# current write weight, a list of (batch, mem_size) array
w_write_t = [T.matrix('w_write_t' + str(h)) for h in range(num_writes)]







# Below are the body of the step-wise function
# Step 1 : Initialize the read heads and write heads
read_heads = [ReadHead(number=h) for h in range(num_reads)]
write_heads = [WriteHead(number=h) for h in range(num_writes)]

# Step 2 : Read from the matrix
read_prev = []












