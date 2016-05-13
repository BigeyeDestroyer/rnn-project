import theano.tensor as T
import numpy
import theano
import sys
sys.path.append('..')
from common.utils import *
from common import mnist_loader
from layers.ram import RAM

# images and labels are the inputs
dataset = mnist_loader.read_data_sets("data")
images, labels = dataset.train.next_batch(batch_size=16)
locs = numpy.random.uniform(low=-1, high=1, size=(images.shape[0], 2))
print images.shape
print labels.shape

# images and locs as inputs


# img : (batch_size, 784)
# normLoc : (batch_size, 2)
img_batch = T.matrix('img')
normLoc = T.matrix('loc')
y = T.ivector('label')


model = RAM(img_batch=img_batch, normLoc=normLoc, y=y)

fn_cost = theano.function(inputs=[img_batch, normLoc, y],
                          outputs=[model.cost])

cost = fn_cost(images, locs, labels)[0]
print type(cost)
print cost






"""
fn_sample = theano.function(inputs=[img_batch, normLoc],
                            outputs=[l, l_sampled, c, h])

l_mean, l_s, c_out, h_out = fn_sample(images, locs)

print type(l_mean)
print l_mean.shape
print l_mean[0, 0, :]

print type(l_s)
print l_s.shape
print l_s[0, 0, :]

print type(c_out)
print c_out.shape

print type(h_out)
print h_out.shape """







