from code_backup.model_train import *

def pred_error(model, prepare_data, data, iterator):
    #Just compute the error
    #f_pred: Theano fct computing the prediction
    #prepare_data: usual prepare_data for that dataset.
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        valid_err += model.error(x, mask, y)
    valid_err /= len(iterator)

    return valid_err



""" params for RNN model
"""
n_words = 100000
in_size = 128
out_size = 2
hidden_size = [128]  # this parameter must be a list

print 'Building the model ...'
model = RNN(n_words, in_size, out_size, hidden_size)

""" params for training
"""
patience = 10
max_epochs = 500
dispFreq = 10
lr = 0.0001
validFreq = 500
saveFreq = 1000
maxlen = None
batch_size = 256
valid_batch_size = 64
dataset = 'imdb'
test_size = 500
saveto = 'lstm_model.npz'

""" load the data
"""
print 'Loading data ...'
train_set, valid_set, test_set = load_imdb(n_words=n_words,
                                           valid_portion=0.1,
                                           maxlen=maxlen)

""" optimization
"""
print 'optimization ...'
valid_shuffle = get_minibatches_idx(len(valid_set[0]), valid_batch_size)
test_shuffle = get_minibatches_idx(len(test_set[0]), valid_batch_size)

print('%d train examples' % len(train_set[0]))
print('%d valid examples' % len(valid_set[0]))
print('%d test examples' % len(test_set[0]))

history_errs = []
best_p = None
bad_counter = 0

if validFreq == -1:
    validFreq = len(train_set[0]) // batch_size
if saveFreq == -1:
    saveFreq = len(train_set[0]) // batch_size

uidx = 0  # the number of update done
estop = False  # early stop
start_time = time.time()

try:
    for eidx in range(max_epochs):
        n_samples = 0

        train_shuffle = get_minibatches_idx(len(train_set[0]), batch_size, shuffle=True)

        for _, train_index in train_shuffle:
            uidx += 1

            x = [train_set[0][t] for t in train_index]
            y = [train_set[1][t] for t in train_index]


            x, mask, y = prepare_data(x, y)
            n_samples += x.shape[1]

            cost = model.train(x, mask, y, lr)

            if numpy.isnan(cost) or numpy.isinf(cost):
                print('bad cost detected: ', cost)
                break

            if numpy.mod(uidx, dispFreq) == 0:
                print('Epoch%d, Update%d, Cost%.6f' % (eidx, uidx, cost))
                #print('Epoch', eidx, ' Update', uidx, ' Cost ', cost)

            if saveto and numpy.mod(uidx, saveFreq) == 0:
                print('Saving...')

            if numpy.mod(uidx, validFreq) == 0:
                train_err = pred_error(model, prepare_data, train_set, train_shuffle)
                valid_err = pred_error(model, prepare_data, valid_set, valid_shuffle)
                test_err = pred_error(model, prepare_data, test_set, test_shuffle)

                history_errs.append([valid_err, test_err])

                print(('Train ', train_err, 'Valid ', valid_err,
                       'Test ', test_err))


except KeyboardInterrupt:
    print('Training interupted ...')

end_time = time.time()
print('The code run for %d epochs, with %f sec/epochs' % (
    (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))