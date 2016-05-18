import theano
import theano.tensor as T
import numpy
from common.utils import *


""" Below are the params for class controller
"""
input_size = 8
output_size = 8
mem_size = 128
mem_width = 20
layer_sizes = [100]
num_heads = 1

x_t = T.matrix('x_t')  # tensor variable, with size (batch, input_size)
# list of tensor variables, each with size (batch, mem_size)
read_tm1_list = [T.matrix('read_tm1_%d' % h) for h in range(num_heads)]
# list of tensor variables, each with size (batch, layer_sizes[l])
h_tm1_list = [T.matrix('h_tm1_%d' % l) for l in len(layer_sizes)]
# list of tensor variables, each with size (batch, layer_sizes[l])
c_tm1_list = [T.matrix('c_tm1_%d' % l) for l in len(layer_sizes)]

""" Below are the class body
"""

# layers is with size [[W0, b0], [W1, b1], ..., [Wn, bn]]
layers = []
# params is with size [W0, b0, W1, b1, ..., Wn, bn]
params = []

h_t_list = []  # each LSTM layer's h
c_t_list = []  # each LSTM layer's c


for i in xrange(len(layer_sizes) + 1):
    if i == 0:
        layers.append([glorot_uniform(shape=(input_size, layer_sizes[i]),
                                           name='W_hidden_%d' % i),
                            init_bias(size=layer_sizes[i], name='b_hidden_%d' % i)])
        params.extend(layers[-1])
    elif 0 < i < len(layer_sizes):
        layers.append([glorot_uniform(shape=(layer_sizes[i - 1], layer_sizes[i]),
                                           name='W_hidden_%d' % i),
                            init_bias(size=layer_sizes[i], name='b_hidden_%d' % i)])
        params.extend(layers[-1])
    else:
        layers.append([glorot_uniform(shape=(layer_sizes[i - 1], output_size),
                                           name='W_hidden_%d' % i),
                            init_bias(size=output_size, name='b_hidden_%d' % i)])
        params.extend(layers[-1])