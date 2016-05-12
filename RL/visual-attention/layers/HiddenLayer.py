__author__ = 'ray'
import theano.tensor as T
import numpy
import theano
class HiddenLayer(object):
    def __init__(self, rng, prefix, input, n_in, n_out, W=None, b=None,
                 activiation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and
        have sigmoidal activiation function. Weight matrix W is of shape (n_in, n_out)
        and the bias vector b is of shape (n_out,).

        NOTE: The nonlinearity used here is tanh

        Hidden unit activiation is given by: tanh(dot(input, W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activiation: theano.OP or function
        :param activiation: Non Linearity to be applied in the hidden layer
        """
        self.input = input
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ),
                dtype=theano.config.floatX
            )
            if activiation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=prefix + '#W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=prefix + '#b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activiation is None
            else activiation(lin_output)
        )

        # parameters of the model
        self.params = [self.W, self.b]
