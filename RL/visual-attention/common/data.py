from __future__ import print_function
import six.moves.cPickle as pickle

import gzip
import os

import numpy
import theano


def seq2seqs(s, l):
    """
    This function cuts a whole sequence into
    sequences with length 'l' and with no overlap

    Parameters
    ----------
    :type s: numpy array or list with
               size (n, )
    :param s: the sequence to be cut

    :type l: int
    :param l: length of the subsequence
    """
    seqs = []
    char_start = 0
    n = len(s)
    for i in range(n // l):
        seqs.append(s[char_start: char_start + l])
        char_start += l
    if char_start != n:
        seqs.append(s[char_start:])

    return seqs


def seq2seqs_persistent(s, l):
    """
    This function cuts a whole sequence into
    sequences with length 'l' and with overlap:
    the previous sequence's endding char is
    the latter sequence's beginning char

    Parameters
    ----------
    :type s: numpy array or list with
               size (n, )
    :param s: the sequence to be cut

    :type l: int
    :param l: length of the subsequence

    :returns
    :type seqs: list of lists or numpy arrays each with size (l, )
    :param seqs: the subsequences processed
    """
    seqs = []
    char_start = 0
    n = len(s)

    # end of the previous sequence is
    # the beginning of the latter one
    for i in range(n // l):
        seqs.append(s[char_start: char_start + l + 1])
        char_start += l
    if char_start != n:
        seqs.append(s[char_start:])

    return seqs


def get_dataset_file(dataset, default_dataset='/home/trunk/disk4/lurui/imdb'):
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


def preprocess_text8(data_path, length,
                     train_portion, valid_portion):
    """
    This function preprocesses the 'text8' dataset
    and cuts them into non-overlapping subsequences

    Parameters
    ----------
    :type data_path: string
    :param data_path: path to load the 'text8' raw data

    :type length: int
    :param length: length of the subsequences to be generated

    :type train_portion: float
    :param train_portion: train set portion

    :type valid_portion: float
    :param valid_portion: valid set portion
    """
    # directory to store the reorganized data
    save_dir = os.path.join(os.path.split(data_path)[0],
                            'text8seqs')

    data = open(data_path, 'r').read()  # total data we have
    chars = list(set(data))  # dictionary of our data

    data_size = len(data)
    chars_size = len(chars)

    # convert the chars into idx
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}

    seq = numpy.zeros((len(data), ), dtype='int8')
    for i in range(len(data)):
        seq[i] = char2idx[data[i]]

    seqs = seq2seqs(seq, length)
    # shuffle the sequences
    sidx = numpy.random.permutation(len(seqs))
    seqs = [seqs[i] for i in sidx]

    # split data into train, valid and test
    num_train = int(numpy.floor(len(seqs) * train_portion))
    num_valid = int(numpy.floor(len(seqs) * valid_portion))

    train_set = [seqs[i] for i in range(0, num_train)]
    valid_set = [seqs[i] for i in range(num_train, num_train + num_valid)]
    test_set = [seqs[i] for i in range(num_train + num_valid, len(seqs))]

    # save the reorganized data to the same directory
    # with that of the original loaded data
    numpy.save(save_dir, [train_set, valid_set, test_set])


def preprocess_text8_persistent(data_path='text8', length=180,
                                length_valid = 1000, train_portion = 0.9,
                                valid_portion = 0.05):
    """
    This function preprocess the 'text8' dataset
    and cuts training set into non-overlap subsequences
    , cuts valid and test set into overlap subsequences

    Parameters
    ----------
    :type length: int
    :param length: length for training subsequences

    :type length_valid: int
    :param length_valid: length for valid and test subsequences
    """
    data_path = get_dataset_file(data_path)
    # directory to store the reorganized data
    save_dir = os.path.join(os.path.split(data_path)[0],
                            'text8seqs_mid')

    data = open(data_path, 'r').read()  # total data we have
    chars = list(set(data))  # dictionary of our data

    # convert the chars into idx
    char2idx = {ch: i for i, ch in enumerate(chars)}

    seq = numpy.zeros((len(data), ), dtype='int8')
    for i in range(len(data)):
        seq[i] = char2idx[data[i]]

    train_len = int(numpy.floor(len(seq) * train_portion))
    valid_len = int(numpy.floor(len(seq) * valid_portion))

    # split the whole 100M data into
    # train, valid and test
    train_seq = seq[: train_len]
    valid_seq = seq[train_len: train_len + valid_len]
    test_seq = seq[train_len + valid_len:]

    train_seqs = seq2seqs(train_seq, length)
    valid_seqs = seq2seqs_persistent(valid_seq, length_valid)
    test_seqs = seq2seqs_persistent(test_seq, length_valid)

    # train_seqs: list of (500000, ) ndarrays with size (180, )
    # valid_seqs: list of (5000, ) ndarrays with size (1001, )
    #             and the last one (1000, )
    # test_seqs: list of (5000, ) ndarrays with size (1001, )
    #             and the last one (1000, )
    numpy.save(save_dir, [train_seqs, valid_seqs, test_seqs])


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


def load_text8(path='text8seqs.npy'):
    """
    This function loads the 'text8' dataset

    Parameters
    ----------
    :type path: str
    :param path: directory to load
                 the '*.npy' data
    """
    path = get_dataset_file(path)

    data = numpy.load(path)
    train_set = data[0]
    valid_set = data[1]
    test_set = data[2]

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

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


def prepare_text8(seqs):
    """
    This function prepare the 'text8' dataset.
    Or specially for those don't have labels

    Parameters
    ----------
    :type seqs: list or numpy with size (n_samples, len)
    :param seqs: input data to be reorganized
    """
    seqs = list(seqs)
    n_samples = len(seqs)
    # for a sequence 's' with length n
    # data  : s[0] ~ s[n - 2]
    # label : s[1] ~ s[n - 1]
    lengths = [len(s) - 1 for s in seqs]
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int8')
    labels = numpy.zeros((maxlen, n_samples)).astype('int8')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
            x[:lengths[idx], idx] = s[0: lengths[idx]]
            labels[:lengths[idx], idx] = s[1: lengths[idx] + 1]
            x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    This function shuffles the samples at the
    beginning of each iteration, this is the
    modified version

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

    # The modified part, we don't make use of the residual
    """
    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
    """

    # zipped (index, list) pair
    return zip(range(len(minibatches)), minibatches)
