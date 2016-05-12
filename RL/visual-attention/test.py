from layers.glimpseSensor import glimpseSensor
import common.mnist_loader as mnist_loader
import numpy
import theano.tensor as T
import theano

import time
import math

dataset = mnist_loader.read_data_sets("data")
images, labels = dataset.train.next_batch(batch_size=16)
locs = numpy.random.uniform(low=-1, high=1, size=(images.shape[0], 2))
print images.shape
print labels.shape


img = T.matrix('img')
loc = T.matrix('loc')
# batch_size = T.iscalar('batch_size')
model = glimpseSensor(layer_id=str(0), batch_size=16,
                      img=img, normLoc=loc)
zoom = model.zooms
fn_zooms = theano.function(inputs=[img, loc], outputs=[zoom])

y = fn_zooms(images, locs)[0]
print type(y)
print y.shape