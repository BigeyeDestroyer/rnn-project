import theano
import theano.tensor as T
import numpy
from head import *
from controller import *
from common.utils import *
from memory import *
from ntm_cell import *
import scipy
import operator
from copy_task import *


"""
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



""" Test read head
batch_size = 5
number = 0
last_dim = 10
mem_size = 6
mem_width = 10
shift_width = 3
similarity = cosine_sim
is_write = False


model = Head(is_write=is_write, last_dim=last_dim,
             mem_size=mem_size, mem_width=mem_width,
             shift_width=shift_width)


M_tm1 = T.tensor3('M_tm1')  # with size (mem_size, mem_width)
w_tm1 = T.matrix('w_tm1')  # with size (batch_size, mem_size)
last_hidden = T.matrix('last_hidden')  # with size (batch, last_dim)
w_t_target = T.matrix('w_t_target')  # with size (batch_size, mem_size)

w_t, read_t = model.step(M_tm1=M_tm1, w_tm1=w_tm1,
                                   last_hidden=last_hidden, batch_size=batch_size)
cost = T.mean((w_t - w_t_target) ** 2)

gparams = []
for param in model.params:
    gparams.append(T.grad(cost, param))

updates = []
for p, gp in zip(model.params, gparams):
    updates.append((p, p - 0.1 * gp))

fn_train = theano.function(inputs=[M_tm1, w_tm1, last_hidden, w_t_target],
                           outputs=cost, updates=updates)
fn_output = theano.function(inputs=[M_tm1, w_tm1, last_hidden],
                            outputs=w_t)

# ndarray inputs
M_in = numpy.random.randn(batch_size, mem_size, mem_width)

w_in = numpy.random.rand(batch_size, mem_size)
w_in = w_in / numpy.reshape(numpy.sum(w_in, axis=1), (w_in.shape[0], 1))

w_out_target = numpy.random.rand(batch_size, mem_size)
w_out_target = w_out_target / numpy.reshape(
    numpy.sum(w_out_target, axis=1), (w_out_target.shape[0], 1))

last_in = numpy.random.randn(batch_size, last_dim)

print model.W_key.get_value()
# call the functions
cost_output = fn_train(M_in, w_in, last_in, w_out_target)
output = fn_output(M_in, w_in, last_in)

print model.W_key.get_value().shape
print model.W_key.get_value()
"""

"""
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




eps = 1e-12
input_dim = 8
output_dim = 8
mem_size = 3
mem_width = 2
layer_sizes = [5]
num_reads = 1
batch_size = 5
num_writes = 1
shift_width = 3

# Below are the body of the step function


x_t = T.matrix('x_tm1')  # current input, (batch, input_size)


model = NTMCell(input_dim=input_dim, output_dim=output_dim,
                mem_size=mem_size, mem_width=mem_width,
                layer_sizes=layer_sizes, num_reads=num_reads,
                batch_size=batch_size, num_writes=num_writes,
                shift_width=shift_width)
output_t = model.step(x_t)
output_target = T.matrix('output_target')
cost = T.mean((output_t - output_target) ** 2)

params = []
for idx in range(num_writes):
    params.append(model.memory.write_heads[idx].W_gamma)

gparams = []
for param in params:
    gparams.append(T.grad(cost, param))

updates = []
for p, gp in zip(params, gparams):
    updates.append((p, p - 0.1 * gp))


#w_read_t_list = state['w_read']
#w_write_t_list = state['w_write']
#read_t_list = state['read']
#c_t_list = state['c']
#h_t_list = state['h']


fn_test = theano.function(inputs=[x_t],
                          outputs=output_t)
fn_train = theano.function(inputs=[x_t, output_target],
                           outputs=cost)

x_data = numpy.random.randn(batch_size, input_dim)
out_target = numpy.random.randn(batch_size, output_dim)

out_data = fn_test(x_data)
cost_out = fn_train(x_data, out_target)

print type(out_data)
print out_data.shape
print out_data

print cost_out


#print w_read_out.shape
#print w_write_out.shape
#print read_out.shape
#print c_out.shape
#print h_out.shape



""" count params
def parameter_count(self):
        import operator
        params = self.__dict__['params']
        count = 0
        for p in params.values():
            shape = p.get_value().shape
            if len(shape) == 0:
                count += 1
            else:
                count += reduce(operator.mul, shape)
        return count

import operator
model = NTMCell()
params = model.params
count = 0
for p in params:
    shape = p.get_value().shape
    if len(shape) == 0:
        count += 1
    else:
        count += reduce(operator.mul, shape)
print count
"""










