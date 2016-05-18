from common.utils import *
import theano
import theano.tensor as T
import numpy
from head import *
from controller_feedforward import *
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
x_t = T.matrix('x_tm1')  # current input, (batch, input_size)
M_tm1 = T.matrix('M_tm1')  # previous memory, (mem_size, mem_width)

# previous read weight, a list of (batch, mem_size) array
w_read_tm1 = [T.matrix('w_read_tm1' + str(h)) for h in range(num_reads)]
# previous write weight, a list of (batch, mem_size) array
w_write_tm1 = [T.matrix('w_write_tm1' + str(h)) for h in range(num_writes)]


# Below are the body of the step-wise function
# Step 1 : Initialize the read heads and write heads
read_heads = [ReadHead(number=h) for h in range(num_reads)]
write_heads = [WriteHead(number=h) for h in range(num_writes)]


# Step 2 : Read from the matrix
def read_memory(M_tm1, w_tm1):
    """
    :type M_tm1: theano variable, with size (mem_size, mem_width)
    :param M_tm1: memory matrix at time t - 1

    :type w_tm1: theano variable, with size (batch_size, mem_size)
    :param w_tm1: head's weight at time t - 1

    :return: with size (batch_size, mem_width)
    """
    return T.dot(w_tm1, M_tm1)

# read_t : a list of (batch_size, mem_width) ndarrays
read_t = [read_memory(M_tm1=M_tm1, w_tm1=w_read_tm1[h])
          for h in range(num_reads)]















