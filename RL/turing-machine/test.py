import theano
import theano.tensor as T
import numpy
from head import *
from controller import *
from common.utils import *
import scipy


""" Test controller
input_size = 8
output_size = 8
mem_size = 128
mem_width = 20
layer_sizes = [100]

X = T.matrix('X')  # with size (batch, input_size)
read_input = T.matrix('read_input')  # with size (batch, mem_width)

model = ControllerFeedforward()
fin_hidden, output = model.step_controller(X, read_input)

fn_step_controller = theano.function(inputs=[X, read_input],
                                     outputs=[fin_hidden, output])

data = numpy.random.randn(5, input_size)
data_read = numpy.random.randn(5, mem_width)

fin_out, out = fn_step_controller(data, data_read)
print type(fin_out)
print fin_out.shape

print type(out)
print out.shape
"""

# Test readhead
batch_size = 5
number = 0
input_size = 100
mem_size = 128
mem_width = 20
shift_width = 3
x_t = T.matrix('x_t')  # with size (batch, input_size)
w_tm1 = T.matrix('w_tm1')  # with size (batch, mem_size)
M_t = T.matrix('M_t')  # with size (mem_size, mem_width)

model = WriteHead()
w_t, erase_t, add_t = model.step(x_t, w_tm1, M_t)
fn_out = theano.function(inputs=[x_t, w_tm1, M_t],
                         outputs=[w_t, erase_t, add_t])

data_x = numpy.random.randn(batch_size, input_size)
data_w = numpy.random.rand(batch_size, mem_size)
data_w = numpy.exp(data_w) / numpy.reshape(numpy.sum(numpy.exp(data_w), axis=1), (batch_size, 1))
data_M = numpy.random.randn(mem_size, mem_width)

output, out_erase, out_add = fn_out(data_x, data_w, data_M)

print type(output)
print output.shape
print numpy.sum(output, axis=1)

print type(out_erase)
print out_erase.shape

print type(out_add)
print out_add.shape


"""
number = 0
input_size = 100
mem_size = 128
mem_width = 20
shift_width = 3
X = T.matrix('X')
model = WriteHead()
key, beta, g, shift, gamma, erase, add = model.step_writehead(X)
fn_step_write = theano.function(inputs=[X],
                                outputs=[key, beta, g, shift, gamma, erase, add])

data = numpy.random.randn(5, input_size)
key_data, beta_data, g_data, shift_data, \
gamma_data, erase_data, add_data = fn_step_write(data)

print type(key_data)
print key_data.shape

print type(beta_data)
print beta_data.shape

print type(g_data)
print g_data.shape

print type(shift_data)
print shift_data.shape

print type(gamma_data)
print gamma_data.shape

print type(erase_data)
print erase_data.shape

print type(add_data)
print add_data.shape
"""










