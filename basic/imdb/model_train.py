import time

from common.data import *
from layers.rnn import *


def pred_error(model, prepare_data, data, iterator):
    """
    This function computes the prediction error
    on the given 'data' dataset

    Parameters
    ----------
    :param model: the trained deep model
    :param prepare_data: function handle, to reorganize the data
    :param data: list of lists, each list meaning a sentence
    :param iterator: returned by the above function
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        valid_err += model.error(x, mask, y)
    valid_err /= len(iterator)

    return valid_err


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    This function shuffles the samples at the
    beginning of each iteration

    Parameters
    ----------
    :type n: int
    :param n: number of samples

    :type minibatch_size: int
    :param minibatch_size:

    :type shuffle: bool
    :param shuffle: whether to shuffle the samples
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    # zipped (index, list) pair
    return zip(range(len(minibatches)), minibatches)


def save_params(file_name, params):
    """
    This function stores params into an '*.h5' file

    Parameters
    ----------
    :type file_name: string
    :param file_name: file name to store the params

    :type params: dict
    :param params: parameters to be stored
    """
    f = h5py.File(file_name)
    for pname, pvalue in params.iteritems():
        f[pname] = pvalue
    f.close()

""" load the data
"""
print 'Loading data ...'
# train_set = (train_set_x, train_set_y)
# valid_set = (valid_set_x, valid_set_y)
# test_set = (test_set_x, test_set_y)
train_set, valid_set, test_set = load_imdb(n_words=100000,
                                           valid_portion=0.1,
                                           maxlen=None)

""" params for RNN model
"""
n_words = 100000
in_size = 128
out_size = 2
hidden_size = [128]  # this parameter must be a list
cell = 'gru'
optimizer = 'rmsprop'
drop_ratio = 0.5

""" params for training
"""
lr = 0.0001
batch_size = 256
valid_batch_size = 64
test_size = 10000

dispFreq = 10  # print each 10 mini-batch
validFreq = 500
saveto = '/home/trunk/disk4/lurui/models/gru_imdb'  # directory of the trained model

max_epochs = 500

# suppose the beginning patience if 10% of the total epochs
patience = (max_epochs / 10) * (len(train_set[0]) / batch_size)

""" Codes below should be the body of function 'model_train'
"""

""" preprocess of the inputs
"""
# 1. redefine the model directory
# if 'saveto' is not a directory, we simply make it
# pointing to the model folder under the same parent folder
root_dir, model_dir = os.path.split(saveto)
if root_dir == '':
    saveto = os.path.join(
        os.path.split(os.path.split(__file__)[0])[0],
        'model'
    )

# 2. select number of test samples according to the 'test_size'
if test_size > 0:
    # The test set is sorted by size, but we want to keep random
    # size example.  So we must select a random selection of the
    # examples.
    idx = numpy.arange(len(test_set[0]))
    numpy.random.shuffle(idx)
    idx = idx[:test_size]
    test_set = ([test_set[0][n] for n in idx], [test_set[1][n] for n in idx])

""" store params related to the model
"""
model_options = dict()
model_options['n_words'] = n_words
model_options['in_size'] = in_size
model_options['hidden_size'] = hidden_size
model_options['out_size'] = out_size
model_options['cell'] = cell
model_options['optimizer'] = optimizer
model_options['drop_ratio'] = drop_ratio

""" build the model
"""
print 'Building the model ...'
model = RNN(n_words=n_words, in_size=in_size, out_size=out_size,
            hidden_size=hidden_size, cell=cell,
            optimizer=optimizer, p=drop_ratio)

""" optimization
"""
print 'Optimization ...'
valid_shuffle = get_minibatches_idx(len(valid_set[0]), valid_batch_size)
test_shuffle = get_minibatches_idx(len(test_set[0]), valid_batch_size)

print('%d train examples' % len(train_set[0]))
print('%d valid examples' % len(valid_set[0]))
print('%d test examples' % len(test_set[0]))

# this means that every epoch, we compute
# validation error and save the model
if validFreq == -1:
    validFreq = len(train_set[0]) // batch_size

history_errs = []  # error for train, valid and test every validFreq
history_train = []  # cost and error for mini-batch training

best_validation_err = numpy.inf
patience_increase = 2  # if an obvious error decrease detected,
                       # we set patience by this multiplication
improvement_threshold = 0.995

epoch = 0  # current epoch number
iters = 0  # number of mini-batches we've gone through
estop = False  # whether stop in the early stop training case
start_time = time.time()

try:
    while (epoch < max_epochs) and (not estop):
        epoch += 1
        n_samples = 0

        train_shuffle = get_minibatches_idx(len(train_set[0]), batch_size, shuffle=True)

        # here, the train_index is a list
        for _, train_index in train_shuffle:
            iters += 1  # start a mini-batch training

            x = [train_set[0][t] for t in train_index]
            y = [train_set[1][t] for t in train_index]

            # x    : (maxlen, mini_batch)
            # mask : (mini_batch, )
            # y    : (mini_batch, )
            x, mask, y = prepare_data(seqs=x, labels=y)
            n_samples += x.shape[1]

            cost = model.train(x, mask, y, lr)
            err = model.error(x, mask, y)

            if numpy.isnan(cost) or numpy.isinf(cost):
                print('bad cost detected: ', cost)
                break

            if numpy.mod(iters, dispFreq) == 0:
                print('Epoch%d, Iters%d, Cost: %.4f, Error: %.4f%%'
                      % (epoch, iters, cost, err * 100.))
                history_train.append([epoch, iters, cost, err])

            if numpy.mod(iters, validFreq) == 0:
                train_err = pred_error(model=model, prepare_data=prepare_data,
                                       data=train_set, iterator=train_shuffle)
                valid_err = pred_error(model=model, prepare_data=prepare_data,
                                       data=valid_set, iterator=valid_shuffle)
                test_err = pred_error(model=model, prepare_data=prepare_data,
                                      data=test_set, iterator=test_shuffle)

                history_errs.append([train_err, valid_err, test_err])
                print('Errs. Train: %.4f, Valid: %.4f, Test: %.4f' %
                      (train_err, valid_err, test_err))

                # suppose we get best validation error until now
                if valid_err < best_validation_err:
                    if valid_err < best_validation_err * improvement_threshold:
                        patience = max(patience, iters * patience_increase)

                    best_validation_err = valid_err
                    # save the model
                    model_file = saveto + '/model.h5'
                    if os.path.isfile(model_file):
                        os.system('rm ' + model_file)
                    print('Saving model at iter%d ...' % iters)
                    model.save_to_file(model_file)

                    # save the errors
                    err_file = saveto + '/err.npz'
                    numpy.savez(err_file, history_errs=history_errs,
                                history_train=history_train)

                    # save model options
                    model_option_file = saveto + '/model.pkl'
                    pickle.dump(model_options, open(model_option_file, 'wb'), -1)
                    print('Done.')


            if patience <= iters:
                estop = True
                break

except KeyboardInterrupt:
    print('Training interupted ...')
end_time = time.time()

# after interrupt, we save the model
# save the model
model_file = saveto + '/model_final.h5'
if os.path.isfile(model_file):
    os.system('rm ' + model_file)
print('Saving model at iter%d ...' % iters)
model.save_to_file(model_file)

# save the errors
err_file = saveto + '/err_final.npz'
numpy.savez(err_file, history_errs=history_errs,
            history_train=history_train)

# save model options
model_option_file = saveto + '/model_final.pkl'
pickle.dump(model_options, open(model_option_file, 'wb'), -1)
print('Done.')


print('The code run for %d epochs, with %f sec/epochs' % (
    (iters + 1), (end_time - start_time) / (1. * (iters + 1))))
print('Training took %.1fs' % (end_time - start_time))



