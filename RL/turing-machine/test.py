import theano
import theano.tensor as T
import numpy
from head import *
from controller import *
from common.utils import *


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

""" Test readhead
number = 0
input_size = 100
mem_size = 128
mem_width = 20
shift_width = 3
X = T.matrix('X')  # with size (batch, input_size)

model = ReadHead()
key, beta, g, shift, gamma = model.step_readhead(X=X)
fn_out = theano.function(inputs=[X],
                         outputs=[key, beta, g, shift, gamma])

data = numpy.random.randn(5, input_size)
key_data, beta_data, g_data, shift_data, gamma_data = fn_out(data)

print type(key_data)
print key_data.shape

print type(beta_data)
print beta_data.shape

print type(g_data)
print g_data.shape

print type(shift_data)
print shift_data.shape

print type(gamma_data)
print gamma_data.shape """

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






