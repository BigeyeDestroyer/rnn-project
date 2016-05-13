import sys
sys.path.append('..')
from common.utils import *
from common import mnist_loader
from layers.ram import RAM

# images and labels are the inputs
dataset = mnist_loader.read_data_sets("data")

# images and locs as inputs
lr = 0.01
print('Building model ...')
model = RAM()

print('Loading data ...')
images, labels = dataset.train.next_batch(batch_size=128)
locs = numpy.random.uniform(low=-1, high=1, size=(images.shape[0], 2))
locs = locs.astype(numpy.float32)

print('Predicting ...')
cost = model.cost(images, locs, labels)
error = model.error(images, locs, labels)
print('Cost: %f, Error: %f' % (cost, error))

pred_prob = model.pred_prob(images, locs)
pred = model.pred(images, locs)
print pred_prob
print pred
for i in range(100):
    images, labels = dataset.train.next_batch(batch_size=16)
    locs = numpy.random.uniform(low=-1, high=1, size=(images.shape[0], 2))
    locs = locs.astype(numpy.float32)
    cost_train = model.train(images, locs, labels, lr)
    print('Iter %d, training cost %f ...' % (i, cost_train))

"""
cost = model.cost(images, locs, labels)
error = model.error(images, locs, labels)
print('cost: %f, error: %f' % (cost, error))
# [cost, error] = model.cost_and_error(images, locs, labels)
# print('%d-th iter, cost: %f' % (1, cost))

"""






