import theano
import theano.tensor as T
import numpy
from head import Head
from controller import *

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


