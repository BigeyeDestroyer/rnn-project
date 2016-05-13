from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
import theano
import numpy

glimpse = 6
batch_size = 16
cell_size = 256

x_0 = T.matrix('x') # with size (16, 256)
numpy_rng = numpy.random.RandomState(1234)
variance = T.tensor3('var')


def _step(var_t, x_tm1):
    x_t = x_tm1 + var_t
    return x_t

[x], _ = theano.scan(fn=_step,
                     sequences=[variance],
                     outputs_info=[x_0])

fn_sample = theano.function(inputs=[x_0, variance],
                            outputs=[x])

x_out = fn_sample(numpy.random.randn(batch_size, cell_size),
                  numpy_rng.normal(loc=0, scale=0.1,
                                   size=(glimpse, batch_size, cell_size)))[0]

print type(x_out)
print len(x_out)
