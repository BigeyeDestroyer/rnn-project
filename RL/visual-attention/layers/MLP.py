__author__ = 'ray'
import numpy
import HiddenLayer
import theano.tensor as T
import logistic

class MLP(object):
    """Multi-Layer Pereptron Class
    """
    def __init__(self, rng, input, n_in, n_hidden, n_out, y):
        """ Initialize the parameters for the multi-layer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of
        the architecture

        :type n_in: int
        :param n_in: number of input units, the dimension of the space
        in  which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space
        in which the labels lie

        :type y: theano.tensor.TensorType
        :param y: symbolic variable that describes the output of
        the architecture
        """

        # the hidden layer
        self.hiddenLayer = HiddenLayer.HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activiation=T.tanh
        )

        # the logistic regression layer
        self.logRegressionLayer = logistic.LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        # L1 norm
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # L2 norm
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # the negative log likelihood of the MLP is given by
        # the negative log likelihood of logRegression Layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood(y)

        # same with that of the logRegression Layer
        self.errors = self.logRegressionLayer.errors(y)

        # params of the structure are made up of two layers
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # keep track of model input
        self.input = input

