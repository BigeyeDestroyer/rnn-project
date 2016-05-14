from common.utils import *
import h5py
from common.data import *
from layers.ram import RAM
import numpy
import time


def pred_error(model, data, label, iterator):
    """
    This function computes the prediction error
    on the given 'data' dataset

    Parameters
    ----------
    :param model: the trained deep model
    :param data: (n, d) ndarray, each row an mnist image
    :param label: (n, ) ndarray, labels
    :param iterator: returned by 'get_minibatches_idx'
                     one unit in:
                     zip(range(len(minibatches)), minibatches)
    """
    err = 0
    for i, index in iterator:
        if i % 5 == 0:
            print('error for %d-th minibatch ...' % i)
        # index is a list of indexes
        images_batch = data[index, :]
        labels_batch = label[index]
        locs = numpy.random.uniform(low=-1, high=1,
                                    size=(images_batch.shape[0], 2))
        locs_batch = locs.astype(numpy.float32)
        err += model.error(images_batch, locs_batch, labels_batch)
    err /= len(iterator)
    return err


def pred_cost(model, data, label, iterator):
    """
    This function computes the prediction cost
    on the given 'data' dataset

    Parameters
    ----------
    :param model: the trained deep model
    :param data: (n, d) ndarray, each row an mnist image
    :param label: (n, ) ndarray, labels
    :param iterator: returned by 'get_minibatches_idx'
                     one unit in:
                     zip(range(len(minibatches)), minibatches)
    """
    cost = 0
    for i, index in iterator:
        if i % 5 == 0:
            print('cost for %d-th minibatch ...' % i)
        # index is a list of indexes
        images_batch = data[index, :]
        labels_batch = label[index]
        locs = numpy.random.uniform(low=-1, high=1,
                                    size=(images_batch.shape[0], 2))
        locs_batch = locs.astype(numpy.float32)
        cost += model.cost(images_batch, locs_batch, labels_batch)
    cost /= len(iterator)
    return cost

""" load the data
"""
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

""" params for training
"""
lr = 1e-3
batch_size = 128

dispFreq = 10  # print each 10 mini-batch
validFreq = 200  # valid every 200 mini-batch
saveto = 'model'

max_epochs = 300

# suppose the beginning patience if 10% of the total epochs
patience = (max_epochs / 10) * (train_set_x.shape[0] / batch_size)

""" build the model
"""
print('Building the model ...')
model = RAM()


""" Optimization
"""
print('Optimization ...')
valid_shuffle = get_minibatches_idx(valid_set_x.shape[0],
                                    batch_size)

history_cost = []  # cost for train, valid every validFreq
history_err = []  # error for train, valid every validFreq
history_train = []  # cost and error for mini-batch training

best_validation_cost = numpy.inf
patience_increase = 2
improvement_threshold = 0.995

epoch = 0  # current epoch number
iters = 0  # number of mini-batches we've gone through
estop = False  # whether stop in the early stop training case
start_time = time.time()

try:
    while (epoch < max_epochs) and (not estop):
        epoch += 1
        train_shuffle = get_minibatches_idx(train_set_x.shape[0], batch_size, shuffle=True)

        # here, the train_index is a list
        for _, train_index in train_shuffle:
            iters += 1  # start a mini-batch training

            images_batch = train_set_x[train_index, :]

            labels_batch = train_set_y[train_index]
            locs = numpy.random.uniform(low=-1, high=1,
                                        size=(images_batch.shape[0], 2))
            locs_batch = locs.astype(numpy.float32)

            cost = model.train(images_batch, locs_batch, labels_batch, lr)
            err = model.error(images_batch, locs_batch, labels_batch)

            if numpy.isnan(cost) or numpy.isinf(cost):
                print('bad cost detected: ', cost)
                break

            if numpy.mod(iters, dispFreq) == 0:
                print('Epoch%d, Iters%d, Cost: %.4f, Error: %.4f%%'
                      % (epoch, iters, cost, err * 100.))
                history_train.append([epoch, iters, cost, err, cost])

            if numpy.mod(iters, validFreq) == 0:
                train_err = pred_error(model=model, data=train_set_x,
                                       label=train_set_y, iterator=train_shuffle)
                valid_err = pred_error(model=model, data=valid_set_x,
                                       label=valid_set_y, iterator=valid_shuffle)

                train_cost = pred_cost(model=model, data=train_set_x,
                                       label=train_set_y, iterator=train_shuffle)
                valid_cost = pred_cost(model=model, data=valid_set_x,
                                       label=valid_set_y, iterator=valid_shuffle)

                history_err.append([train_err, valid_err])
                print('Errs. Train: %.4f, Valid: %.4f' %
                      (train_err, valid_err))
                history_cost.append([train_cost, valid_cost])
                print('Cost. Train: %.4f, Valid: %.4f' %
                      (train_cost, valid_cost))

                # suppose we get best validation error until now
                if valid_cost < best_validation_cost:
                    if valid_cost < best_validation_cost * improvement_threshold:
                        patience = max(patience, iters * patience_increase)

                    best_validation_cost = valid_cost
                    # save the model
                    model_file = saveto + '/model_multi_scale.h5'
                    if os.path.isfile(model_file):
                        os.system('rm ' + model_file)
                    print('Saving model at iter%d ...' % iters)
                    model.save_to_file(model_file)

                    # save the errors
                    err_file = saveto + '/err_multi_scale.npz'
                    numpy.savez(err_file, history_errs=history_err,
                                history_train=history_train,
                                history_cost=history_cost)

                    print('Done.')

            if patience <= iters:
                estop = True
                break
except KeyboardInterrupt:
    print('Training interupted ...')
end_time = time.time()


# after interrupt, we save the model
# save the model
model_file = saveto + '/model_multi_scale_final.h5'
if os.path.isfile(model_file):
    os.system('rm ' + model_file)
print('Saving model at iter%d ...' % iters)
model.save_to_file(model_file)

# save the errors
err_file = saveto + '/err_multi_scale_final.npz'
numpy.savez(err_file, history_errs=history_err,
            history_train=history_train,
            history_cost=history_cost)
print('Done.')

print('The code run for %d epochs, with %f min/epoch' % (
    (iters + 1), (end_time - start_time) / 60. / (1. * (epoch + 1))))
print('Training took %.1fs' % (end_time - start_time))





