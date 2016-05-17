import theano
import theano.tensor as T
import numpy
from head import *
from controller import *

"""
X = T.matrix('X')
read_input = T.matrix('read_input')
batch_size = 5
input_size = 8
output_size = 8
mem_size = 128
mem_width = 20
layer_sizes = [100, 200, 300]

model = ControllerFeedforward(X=X, read_input=read_input,
                              layer_sizes=layer_sizes)
fn_out = theano.function(inputs=[X, read_input],
                         outputs=[model.fin_hidden, model.output])

x = numpy.random.randn(batch_size, input_size)
read = numpy.random.randn(batch_size, mem_width)
[fin_hid, output] = fn_out(x, read)

print type(fin_hid)
print fin_hid.shape

print type(output)
print output.shape
"""

number = 0
input_size = 100
mem_size = 128
mem_width = 20
shift_width = 3
X = T.matrix('X')  # with size (batch, input_size)

model = ReadHead(X=X)




data = numpy.random.randn(5, input_size)

"""
key, beta, g, shift, gamma = model.head_output
fn_out = theano.function(inputs=[X], outputs=[key, beta, g, shift, gamma])
k, b, gg, s, ga = fn_out(data)
print type(k)
print k.shape

print type(b)
print b.shape

print type(gg)
print gg.shape

print type(s)
print s.shape

print type(ga)
print ga.shape
"""

fn_count = theano.function(inputs=[X], outputs=model.count)
count = fn_count(data)

print type(count)
print count




