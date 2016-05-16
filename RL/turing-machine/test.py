import theano
import theano.tensor as T
import numpy
from head import Head

X = T.matrix('X')
model = Head(X=X)
fn_output = theano.function(inputs=[X],
                            outputs=model.head_output)

a = numpy.random.randn(5, 100)
key, beta, g, shift, gamma, erase, add = fn_output(a)

print type(key)
print key.shape

print type(beta)
print beta.shape

print type(g)
print g.shape

print type(shift)
print shift.shape

print type(gamma)
print gamma.shape

print type(erase)
print erase.shape

print type(add)
print add.shape

