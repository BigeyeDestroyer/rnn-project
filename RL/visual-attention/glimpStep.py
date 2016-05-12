import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import sys
from common.utils import *
import numpy
import theano
import theano.tensor.signal.pool as pool
import theano.tensor.nnet.abstract_conv as upsample
from common import mnist_loader
import h5py

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

# the other parameters used in glimpseSensor
batch_size = 16
mnist_size=28
channels=1
depth=3
minRadius=4
sensorBandwidth=8

loc = ((normLoc + 1) / 2) * mnist_size
loc = T.cast(loc, 'int32')

# img with size (batch, height, width, channels)
img = T.reshape(img_batch, (batch_size, mnist_size, mnist_size, channels))

zooms = []  # zooms of all the images in batch

maxRadius = minRadius * (2 ** (depth - 1))  # radius of the largest zoom
offset = maxRadius

# zero-padding the batch to (batch, h + 2R, w + 2R, channels)
img = T.concatenate((T.zeros((batch_size, maxRadius, mnist_size, channels)), img), axis=1)
img = T.concatenate((img, T.zeros((batch_size, maxRadius, mnist_size, 1))), axis=1)
img = T.concatenate((T.zeros((batch_size, mnist_size + 2 * maxRadius, maxRadius, 1)), img), axis=2)
img = T.concatenate((img, T.zeros((batch_size, mnist_size + 2 * maxRadius, maxRadius, 1))), axis=2)
img = T.cast(img, dtype=theano.config.floatX)

for k in xrange(batch_size):
    imgZooms = []  # zoom for a single image

    # one_img with size (2R + size, 2R + size, 1)
    one_img = img[k, :, :, :]

    for i in xrange(depth):
        # r = minR, 2 * minR, ..., (2^(depth - 1)) * minR
        r = minRadius * (2 ** i)

        d_raw = 2 * r  # patch size to be cropped

        loc_k = loc[k, :]  # location of the k-th glimpse, (2, )
        adjusted_loc = T.cast(offset + loc_k - r, 'int32')  # upper-left corner of the patch

        one_img = T.reshape(one_img, (one_img.shape[0], one_img.shape[1]))

        # Get a zoom patch with size (d_raw, d_raw) from one_image
        zoom = one_img[adjusted_loc[0]: (adjusted_loc[0] + d_raw),
               adjusted_loc[1]: (adjusted_loc[1] + d_raw)]

        if r < sensorBandwidth:  # bilinear-interpolation
            #  here, zoom is a 2D patch with size (2r, 2r)
            zoom_reshape = T.reshape(zoom, (1, 1, zoom.shape[0], zoom.shape[1]))
            zoom_bandwidth = upsample.bilinear_upsampling(zoom_reshape,
                                                          ratio=(sensorBandwidth / r),
                                                          batch_size=1, num_input_channels=1)
            zoom_bandwidth = T.reshape(zoom_bandwidth, (zoom_bandwidth.shape[2],
                                                        zoom_bandwidth.shape[3]))
        elif r > sensorBandwidth:
            zoom_bandwidth = pool.pool_2d(input=zoom,
                                          ds=(r / sensorBandwidth,
                                              r / sensorBandwidth),
                                          mode='average_inc_pad',
                                          ignore_border=True)
        else:
            zoom_bandwidth = zoom

        imgZooms.append(zoom_bandwidth)

    zooms.append(T.stack(imgZooms))

zooms = T.stack(zooms)








fn_reshape = theano.function(inputs=[img_batch, normLoc],
                             outputs=[zooms])

zoom_reshape = fn_reshape(images, locs)[0]

f = h5py.File('zoom.h5')
f['zoom'] = zoom_reshape
f.close()

print type(zoom_reshape)
print zoom_reshape.shape




