import theano.tensor as T
import theano
import numpy as np
import numpy
import cPickle as pickle

import inspect


class Parameters(object):
    def __init__(self):
        self.__dict__['params'] = {}

    def __setattr__(self, key, value):
        params = self.__dict__['params']
        if key not in params:
            params[key] = theano.shared(
                value=numpy.asarray(
                    value,
                    dtype=theano.config.floatX
                ),
                borrow=True,
                name=key
            )
        else:
            print('%s already assigned' % key)
            params[key].set_value(numpy.asarray(
                value,
                dtype=theano.config.floatX
            ))

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getattr__(self, key):
        params = self.__dict__['params']
        return params[key]

    def __getitem__(self, key):
        return self.__getattr__(key)

    def remove(self, key):
        del self.__dict__['params'][key]

    def values(self):
        """
        :return: all the shared variables in
                 the dictionary 'params'
        """
        params = self.__dict__['params']
        return params.values()

    def save(self, filename):
        params = self.__dict__['params']
        with open(filename, 'wb') as f:
            pickle.dump({p.name: p.get_value() for p in params.values()}, f, 2)

    def load(self, filename):
        params = self.__dict__['params']
        loaded = pickle.load(open(filename, 'rb'))
        for k in params:
            if k in loaded:
                params[k].set_value(loaded[k])
            else:
                print('%s does not exist.' % k)

    def parameter_count(self):
        import operator
        params = self.__dict__['params']
        count = 0
        for p in params.values():
            shape = p.get_value().shape
            if len(shape) == 0:
                count += 1
            else:
                count += reduce(operator.mul, shape)
        return count
