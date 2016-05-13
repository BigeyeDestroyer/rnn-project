from common.utils import *
import h5py
from common.data import *
from layers.ram import RAM

print('Loading data ...')
f = h5py.File('data/train.h5')
train_set_x = numpy.array(f['images'])  # (60000, 784)
train_set_y = numpy.array(f['labels'])  # (60000, )
f.close()

f = h5py.File('data/test.h5')
valid_set_x = numpy.array(f['images'])  # (10000, 784)
valid_set_y = numpy.array(f['labels'])  # (10000, )
f.close()
print('We have %d train samples, %d valid samples' %
      (train_set_x.shape[0], valid_set_x.shape[0]))

batch_size = 128
lr = 1e-3
max_iters = 1000000

valid_shuffle = get_minibatches_idx(valid_set_x.shape[0], batch_size)
data = valid_set_x
label = valid_set_y

for i, index in valid_shuffle:
    if i % 5 == 0:
        print('error for %d-th minibatch ...' % i)

    # index is a list of indexes
    images_batch = data[index, :]
    labels_batch = label[index]
    locs = numpy.random.uniform(low=-1, high=1,
                                size=(images_batch.shape[0], 2))








