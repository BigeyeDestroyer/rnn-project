import numpy
import theano
import theano.tensor as T
from parameters import Parameters


def initial_weights(*argv):
    """

    :param argv: *argv can receive arbitrary number of
                 parameters, and argv is a tuple
    """
    return numpy.asarray(
        numpy.random.uniform(
            low=-numpy.sqrt(6. / sum(argv)),
            high=numpy.sqrt(6. / sum(argv)),
            size=argv
        ),
        dtype=theano.config.floatX
    )

input_size = 8
output_size = 8
mem_size = 128  # mem_size and mem_width are for the memory matrix
mem_width = 20
layer_sizes = [100]

P = Parameters()
P.W_input_hidden = initial_weights(input_size, layer_sizes[0])
P.W_read_hidden = initial_weights(mem_width, layer_sizes[0])
P.b_hidden_0 = 0. * initial_weights(layer_sizes[0])

hidden_weights = []
for i in xrange(len(layer_sizes) - 1):
    P['W_hidden_%d' % (i + 1)] = initial_weights(layer_sizes[i], layer_sizes[i + 1])
    P['b_hidden_%d' % (i + 1)] = 0. * initial_weights(layer_sizes[i + 1])
    hidden_weights.append((P['W_hidden_%d' % (i + 1)],
                           P['b_hidden_%d' % (i + 1)]))

P.W_hidden_output = 0. * initial_weights(layer_sizes[-1], output_size)
P.b_output = 0. * initial_weights(output_size)


def controller(input_t, read_t):
    prev_layer = hidden_0 = T.tanh(
        T.dot(input_t, P.W_input_hidden) +
        T.dot(input_t, P.W_read_hidden) +
        P.b_hidden_0
    )

    for W, b in hidden_weights:
        prev_layer = T.tanh(T.dot(prev_layer, W) + b)

    fin_hidden = prev_layer
    output_t = T.nnet.sigmoid(
        T.dot(fin_hidden, P.W_hidden_output) + P.b_output)

    return output_t, fin_hidden


