import theano.tensor as T
import numpy
import theano
from common import mnist_loader
import h5py
from layers.glimpseSensor import glimpseSensor

# images and labels are the inputs
dataset = mnist_loader.read_data_sets("data")
images, labels = dataset.train.next_batch(batch_size=16)
locs = numpy.random.uniform(low=-1, high=1, size=(images.shape[0], 2))
print images.shape
print labels.shape


# img : (batch_size, 784)
# normLoc : (batch_size, 2)
img_batch = T.matrix('img')
normLoc = T.matrix('loc')

model = glimpseSensor(img_batch, normLoc)


fn_reshape = theano.function(inputs=[img_batch, normLoc],
                             outputs=[model.zooms])

zoom_reshape = fn_reshape(images, locs)[0]

f = h5py.File('zoom.h5')
f['zoom'] = zoom_reshape
f.close()

print type(zoom_reshape)
print zoom_reshape.shape




