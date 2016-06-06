from __future__ import print_function
import six.moves.cPickle as pickle

import gzip
import os

import numpy
import theano


def get_dataset_file(dataset, default_dataset='/home/trunk/disk4/lurui/imdb/imdb.pkl'):
    """ This function gets the imdb.pkl directory
    """
    data_dir, data_file = os.path.split(dataset)
    if data_dir == '' and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            '..',
            'data',
            dataset
        )
        if os.path.isfile(new_path):
            dataset = new_path
        else:
            dataset = default_dataset

    return dataset


def load_imdb(path='imdb.pkl', n_words=100000, valid_portion=0.1,
              maxlen=None, sort_by_len=True):
    """Loads the dataset, specially for dataset 'imdb'

    :type path: String
    :param path: The path to the dataset IMDB

    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
                    All extra words are set to unknow (1).

    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
                          the validation set.

    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.

    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.
    """

    """ Step 1: read in train data and test data
    """
    path = get_dataset_file(path)

    if path.endswith('.gz'):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_set = pickle.load(f)
    test_set = pickle.load(f)
    f.close()

    """ Step 2: discard those sentences that exceed the max length
    """
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    """ Step 3: split train set into valid set according to the 'valid_portion'
    """
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    """ Step 4: remove words that exceed the most
                frequent indexes given by 'n_words'
    """
    def remove_unk(x):
        # x here is a list of list, meaning a list of sentences
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    test_set_x = remove_unk(test_set_x)
    valid_set_x = remove_unk(valid_set_x)
    train_set_x = remove_unk(train_set_x)

    """ Step 5: sort each dataset's samples according to the
                length of sentences in ascend order
    """
    def len_argsort(seq):
        # seq here is a list of list, meaning a list of sentences
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    test_set = (test_set_x, test_set_y)

    return train_set, valid_set, test_set


def prepare_data(seqs, labels=None, maxlen=None):
    """ Create the matrices from datasets. specially for those have labels

        This pad each sequence to the same length: the
        length of the longest sequence or maxlen

        :param seqs: list of list, lists of sentences,
                     each sentence with different length
        :param labels: list of labels
        :param maxlen: maximum length we use in the experiment

        :return:
        :type x: ndarray with size (maxlen, n_samples)
        :param x: data fed into the rnn

        :type x_mask: ndarray with size (maxlen, n_samples)
        :param x_mask: mask for the data matrix 'x'

        :param labels: list of lables just as the input
    """
    seqs = list(seqs)
    lengths = [len(s) for s in seqs]

    if labels is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    # each column in x corresponding to a sample
    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


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
