import theano
import theano.tensor as T
import numpy
from head import *
from controller_feedforward import *
from controller_lstm import *
from common.utils import *


num_read_heads = 3
num_write_heads = 3
mem_size = 128
mem_width = 20
layer_sizes = [100]

# 1. Build all the read heads and write heads
read_heads = []  # read_heads, list of read heads
for idx in xrange(num_read_heads):
    read_heads.append(Head(idx=idx, last_dim=layer_sizes[-1],
                           is_write=False))

write_heads = []  # write_heads, list of write heads
for idx in xrange(num_write_heads):
    write_heads.append(Head(idx=idx, last_dim=layer_sizes[-1],
                            is_write=True))

M_tm1 = T.matrix('M_tm1')  # with size (mem_size, mem_width)
# list of tensor variables, each with size (batch_size, mem_size)
w_read__tm1_list = [T.matrix('w_read_tm1_%d' % h)
                    for h in range(num_read_heads)]
# list of tensor variables, each with size (batch_size, mem_size)
w_write_tm1_list = [T.matrix('w_write_tm1_%d' % h)
                    for h in range(num_write_heads)]
last_hidden = T.matrix('last_hidden')  # with size (batch_size, last_dim)


w_read__t_list = []
read_t_list = []
# Get the read heads' output
for idx in xrange(num_read_heads):
    w_read_t, read_t = read_heads[idx].step(M_tm1=M_tm1,
                                            w_tm1=w_read__tm1_list[idx],
                                            last_hidden=last_hidden)
    w_read__t_list.append(w_read_t)
    read_t_list.append(read_t)


w_write_t_list = []
erase_t_list = []
add_t_list = []


M_t_erases = []
M_t_adds = []
# Get the write heads' output
for idx in xrange(num_write_heads):
    # erase_t with size (batch_size, mem_width)
    # add_t with size (batch_size, mem_width)
    w_write_t, erase_t, add_t = write_heads[idx].step(M_tm1=M_tm1,
                                                      w_tm1=w_write_tm1_list[idx],
                                                      last_hidden=last_hidden)
    w_write_t_list.append(w_write_t)
    erase_t_list.append(erase_t)
    add_t_list.append(add_t)

    M_t_erases.append()
