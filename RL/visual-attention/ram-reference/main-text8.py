import time
import sys
sys.path.append('..')
from common.data import *
from rnn_step import *
import pickle


def pred_error(model, data, iterator):
    """
    This function computes the prediction error
    on the given 'data' dataset

    Parameters
    ----------
    :param model: the trained deep model
    :param data: list of lists, each list meaning a sentence
    :param iterator: returned by 'get_minibatches_idx'
                     one unit in:
                     zip(range(len(minibatches)), minibatches)
    """
    err = 0
    for _, index in iterator:
        x, mask, y = prepare_text8([data[i] for i in index])
        err += model.error(x, mask, y)
    err /= len(iterator)

    return err


def pred_bpc(model, data, iterator):
    """
    This function computes the prediction bpc
    on the given 'data' dataset

    Parameters
    ----------
    :param model: the trained deep model
    :param data: list of lists, each list meaning a sentence
    :param iterator: returned by 'get_minibatches_idx'
                     one unit in:
                     zip(range(len(minibatches)), minibatches)
    """
    bpc = 0
    for _, index in iterator:
        x, mask, y = prepare_text8([data[i] for i in index])
        bpc += model.bpc(x, mask, y)
    bpc /= len(iterator)

    return bpc






""" load the data
"""
print 'Loading data ...'
# train_set : list of ndarray with size (l, )
# valid_set : list of ndarray with size (l, )
# valid_set : list of ndarray with size (l, )
train_set, valid_set, test_set = load_text8()

""" params for RNN model
"""
n_words = 27  # this param is for embedding's initialization
in_size = 128  # embedding size
out_size = 27  # we totally have 27 chars
hidden_size = [256]  # this parameter must be a list
cell = 'lstm'
optimizer = 'adam'
drop_ratio = 0.5

""" params for training
"""
lr = 0.001
batch_size = 256
valid_batch_size = 256

dispFreq = 10  # print each 10 mini-batch
validFreq = 500
saveto = '/home/trunk/disk4/lurui/models/lstm_text8'  # directory of the trained model

max_epochs = 100

# suppose the beginning patience if 10% of the total epochs
patience = (max_epochs / 10) * (len(train_set) / batch_size)

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
valid_shuffle = get_minibatches_idx(len(valid_set), valid_batch_size)
test_shuffle = get_minibatches_idx(len(test_set), valid_batch_size)

print('%d train examples' % len(train_set))
print('%d valid examples' % len(valid_set))
print('%d test examples' % len(test_set))

# this means that every epoch, we compute
# validation error and save the model
if validFreq == -1:
    validFreq = len(train_set) // batch_size

history_bpc = []  # bpc for train, valid and test every validFreq
history_errs = []  # error for train, valid and test every validFreq
history_train = []  # cost and error for mini-batch training

best_validation_bpc = numpy.inf
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

        train_shuffle = get_minibatches_idx(len(train_set), batch_size, shuffle=True)

        # here, the train_index is a list
        for _, train_index in train_shuffle:
            iters += 1  # start a mini-batch training

            x = [train_set[t] for t in train_index]

            # x    : (maxlen, mini_batch)
            # mask : (maxlen, mini_batch)
            # y    : (maxlen, mini_batch)
            x, mask, y = prepare_text8(seqs=x)
            n_samples += x.shape[1]

            cost = model.train(x, mask, y, lr)
            err = model.error(x, mask, y)
            bpc = model.bpc(x, mask, y)

            if numpy.isnan(cost) or numpy.isinf(cost):
                print('bad cost detected: ', cost)
                break

            if numpy.mod(iters, dispFreq) == 0:
                print('Epoch%d, Iters%d, Cost: %.4f, Error: %.4f%%, Bpc: %.4f'
                      % (epoch, iters, cost, err * 100., bpc))
                history_train.append([epoch, iters, cost, err, bpc])

            if numpy.mod(iters, validFreq) == 0:
                train_err = pred_error(model=model, data=train_set,
                                       iterator=train_shuffle)
                valid_err = pred_error(model=model, data=valid_set,
                                       iterator=valid_shuffle)
                test_err = pred_error(model=model, data=test_set,
                                      iterator=test_shuffle)

                train_bpc = pred_bpc(model=model, data=train_set,
                                     iterator=train_shuffle)
                valid_bpc = pred_bpc(model=model, data=valid_set,
                                     iterator=valid_shuffle)
                test_bpc = pred_bpc(model=model, data=test_set,
                                    iterator=test_shuffle)

                history_errs.append([train_err, valid_err, test_err])
                print('Errs. Train: %.4f, Valid: %.4f, Test: %.4f' %
                      (train_err, valid_err, test_err))
                history_bpc.append([train_bpc, valid_bpc, test_bpc])
                print('Bpc. Train: %.4f, Valid: %.4f, Test: %.4f' %
                      (train_bpc, valid_bpc, test_bpc))

                # suppose we get best validation error until now
                if valid_bpc < best_validation_bpc:
                    if valid_bpc < best_validation_bpc * improvement_threshold:
                        patience = max(patience, iters * patience_increase)

                    best_validation_bpc = valid_bpc
                    # save the model
                    model_file = saveto + '/model.h5'
                    if os.path.isfile(model_file):
                        os.system('rm ' + model_file)
                    print('Saving model at iter%d ...' % iters)
                    model.save_to_file(model_file)

                    # save the errors
                    err_file = saveto + '/err.npz'
                    numpy.savez(err_file, history_errs=history_errs,
                                history_train=history_train,
                                history_bpc=history_bpc)

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
            history_train=history_train,
            history_bpc=history_bpc)

# save model options
model_option_file = saveto + '/model_final.pkl'
pickle.dump(model_options, open(model_option_file, 'wb'), -1)
print('Done.')


print('The code run for %d epochs, with %f min/epoch' % (
    (iters + 1), (end_time - start_time) / 60. / (1. * (epoch + 1))))
print('Training took %.1fs' % (end_time - start_time))



