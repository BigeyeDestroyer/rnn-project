import theano
import theano.tensor as T
import numpy
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


def vector_softmax(vec):
    return T.nnet.softmax(vec.reshape((1, vec.shape[0])))[0]


# Since the input_size is hidden_layers[-1]
# we are actually writing the codes for 'write head'
P = Parameters()
id = 0
input_size = 100
mem_size = 128  # mem_size and mem_width are for the memory matrix
mem_width = 20
shift_width = 3

# 1. Content addressing
P['W_%d_key' % id] = initial_weights(input_size, mem_width)
P['b_%d_key' % id] = 0. * initial_weights(mem_width)
P['W_%d_beta' % id] = 0. * initial_weights(input_size)
P['b_%d_beta' % id] = 0.

# 2. Interpolation
P['W_%d_g' % id] = 0. * initial_weights(input_size)
P['b_%d_g' % id] = 0.

# 3. Convolutional shift
P['W_%d_shift' % id] = initial_weights(input_size, shift_width)
P['b_%d_shift' % id] = 0. * initial_weights(shift_width)

# 4. Sharpening
P['W_%d_gamma' % id] = initial_weights(input_size)
P['b_%d_gamma' % id] = 0.

# 5. Erase and add vector
P['W_%d_erase' % id] = initial_weights(input_size, mem_width)
P['b_%d_erase' % id] = 0. * initial_weights(mem_width)
P['W_%d_add' % id] = initial_weights(input_size, mem_width)
P['b_%d_add' % id] = 0. * initial_weights(mem_width)

x = T.matrix('controller_output')

# 1. Content Addressing: key and beta
key_t = T.dot(x, P['W_%d_key' % id] + P['b_%d_key' % id])
beta_t = T.nnet.softplus(T.dot(x, P['W_%d_beta' % id]) +
                         P['b_%d_beta' % id])

# 2. Interpolation: g
g_t = T.nnet.sigmoid(T.dot(x, P['W_%d_g' % id]) +
                     P['b_%d_g' % id])

# 3. Convolutional Shift: shift_t, with size (batch, shift_width)
# x: theano.tensor with size (batch, input_size)
# shift_x before softmax: with size (batch, shift_width)
# T.nnet.softmax compute the softmax values row-wise
shift_t = T.nnet.softmax(T.dot(x, P['W_%d_shift' % id]) +
                         P['b_%d_shift' % id])
shift_t.name = 'shift_t'

# 4. Sharpening: >= 1
gamma_t = T.nnet.softplus(T.dot(x, P['W_%d_gamma' % id]) +
                          P['b_%d_gamma' % id]) + 1.

erase_t = T.nnet.sigmoid(T.dot(x, P['W_%d_erase' % id]) +
                         P['b_%d_erase' % id])
add_t = T.dot(x, P['W_%d_add' % id]) + P['b_%d_add' % id]







