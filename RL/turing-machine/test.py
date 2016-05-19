import theano
import theano.tensor as T
import numpy
from head import *
from controller_feedforward import *
from controller_lstm import *
from common.utils import *
from memory import *
from ntm_cell import *
import scipy
import operator


""" Test controller
batch_size = 5
input_size = 8
output_size = 8
mem_size = 128
mem_width = 20
layer_sizes = [100, 150]
num_heads = 3

model = ControllerLSTM(layer_sizes=layer_sizes)

x_t = T.matrix('x_t')
read_tm1_list = [T.matrix('read_tm1_%d' % h) for h in xrange(num_heads)]
c_tm1_list = [T.matrix('c_tm1_%d' % l) for l in xrange(len(layer_sizes))]
h_tm1_list = [T.matrix('h_tm1_%d' % l) for l in xrange(len(layer_sizes))]

inputs = [x_t]
inputs.extend(read_tm1_list)
inputs.extend(c_tm1_list)
inputs.extend(h_tm1_list)

c_t_list, h_t_list = model.step(x_t, read_tm1_list, c_tm1_list, h_tm1_list)

outputs = []
outputs.extend(c_t_list)
outputs.extend(h_t_list)

fn_lstm = theano.function(inputs=inputs,
                          outputs=outputs)

data_x = numpy.random.randn(batch_size, input_size).astype(dtype=theano.config.floatX)
read_list = [numpy.random.randn(batch_size, mem_width).astype(dtype=theano.config.floatX) for h in range(num_heads)]
c_list = [numpy.random.randn(batch_size, layer_sizes[l]).astype(dtype=theano.config.floatX) for l in range(len(layer_sizes))]
h_list = [numpy.random.randn(batch_size, layer_sizes[l]).astype(dtype=theano.config.floatX) for l in range(len(layer_sizes))]

data_in = [data_x] + read_list + c_list + h_list


data_out = fn_lstm(data_in[0], data_in[1], data_in[2], data_in[3], data_in[4], data_in[5], data_in[6], data_in[7])  # error occurs here, we should break list data_in

print type(data_out)
print len(data_out)

print type(data_out[0])
print data_out[0].shape

print type(data_out[1])
print data_out[1].shape

print type(data_out[2])
print data_out[2].shape

print type(data_out[3])
print data_out[3].shape
"""



""" Test head
batch_size = 5
number = 0
last_dim = 100
mem_size = 128
mem_width = 20
shift_width = 3
similarity = cosine_sim
is_write = False


model = Head(is_write=is_write)

M_tm1 = T.tensor3('M_tm1')  # with size (mem_size, mem_width)
w_tm1 = T.matrix('w_tm1')  # with size (batch_size, mem_size)
last_hidden = T.matrix('last_hidden')  # with size (batch, last_dim)

w_t, read_t = model.step(M_tm1=M_tm1, w_tm1=w_tm1,
                         last_hidden=last_hidden, batch_size=batch_size)

fn_head = theano.function(inputs=[M_tm1, w_tm1, last_hidden],
                          outputs=[w_t, read_t])

M_in = numpy.random.randn(batch_size, mem_size, mem_width)
w_in = numpy.random.randn(batch_size, mem_size)
last_in = numpy.random.randn(batch_size, last_dim)

w_out, read_out = fn_head(M_in, w_in, last_in)

print type(w_out)
print w_out.shape

print type(read_out)
print read_out.shape
"""

""" Test memory

batch_size = 5
num_read_heads = 2
num_write_heads = 1
mem_size = 128
mem_width = 20
layer_sizes = [100]

"""
"""
M_tm1 = T.tensor3('M_tm1')  # with size (batch_size, mem_size, mem_width)
# list of tensor variables, each with size (batch_size, mem_size)
w_read_tm1_list = [T.matrix('w_read_tm1_%d' % h)
                   for h in range(num_read_heads)]
# list of tensor variables, each with size (batch_size, mem_size)
w_write_tm1_list = [T.matrix('w_write_tm1_%d' % h)
                    for h in range(num_write_heads)]
last_hidden = T.matrix('last_hidden')  # with size (batch_size, last_dim)
"""
"""
model = Memory(batch_size=batch_size, num_read_heads=num_read_heads,
               num_write_heads=num_write_heads, layer_sizes=layer_sizes)

M_t, w_read_t_list, \
w_write_t_list, read_t_list = model.step(M_tm1, w_read_tm1_list,
                                         w_write_tm1_list, last_hidden)
inputs = [M_tm1] + w_read_tm1_list + w_write_tm1_list + [last_hidden]
outputs = [M_t] + w_read_t_list + w_write_t_list + read_t_list

fn_test = theano.function(inputs=inputs,
                          outputs=outputs)

M_data = numpy.random.randn(batch_size, mem_size, mem_width)
w_read1 = numpy.random.randn(batch_size, mem_size)
w_read2 = numpy.random.randn(batch_size, mem_size)
w_write1 = numpy.random.randn(batch_size, mem_size)
last_data = numpy.random.randn(batch_size, layer_sizes[-1])

M_out, w_read1_out, w_read2_out, \
w_write1_out, read1_out, read2_out = fn_test(M_data, w_read1,
                                             w_read2, w_write1,
                                             last_data)
print type(M_out)
print M_out.shape

print w_read1_out.shape

print w_write1_out.shape

print read2_out.shape
"""


""" Test
"""
input_dim = 8
output_dim = 8
mem_size = 128
mem_width = 20
layer_sizes = [100]
num_reads = 1
batch_size = 5
num_writes = 1
shift_width = 3

# Below are the body of the step function


""" state contains: M_tm1, w_read_tm1_list, w_write_tm1_list
                    read_tm1_list, c_tm1_list, h_tm1_list
"""
x_t = T.matrix('x_tm1')  # current input, (batch, input_size)
M_tm1 = T.tensor3('M_tm1')  # previous memory, (mem_size, mem_width)
# previous read weight, a list of (batch, mem_size) array
w_read_tm1_list = [T.matrix('w_read_tm1' + str(h)) for h in range(num_reads)]
# previous write weight, a list of (batch, mem_size) array
w_write_tm1_list = [T.matrix('w_write_tm1' + str(h)) for h in range(num_writes)]
read_tm1_list = [T.matrix('read_tm1' + str(h)) for h in range(num_reads)]
c_tm1_list = [T.matrix('c_tm1_%d' % l) for l in xrange(len(layer_sizes))]
h_tm1_list = [T.matrix('h_tm1_%d' % l) for l in xrange(len(layer_sizes))]

# test
inputs = [x_t] + [M_tm1] + w_read_tm1_list + w_write_tm1_list + \
         read_tm1_list + c_tm1_list + h_tm1_list
model = NTMCell()
M_t, w_read_t_list, w_write_t_list, read_t_list, c_t_list, h_t_list = \
    model.step(x_t, M_tm1, w_read_tm1_list, w_write_tm1_list, read_tm1_list, c_tm1_list, h_tm1_list)
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
