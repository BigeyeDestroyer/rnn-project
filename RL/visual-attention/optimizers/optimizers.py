import theano.tensor as T
import sys
sys.path.append('..')
from common.utils import *


def sgd(params, gparams, lr=0.01, momentum=0., decay=0., nesterov=False):
    """
    Stochastic Gradient Descent with momentum

    Parameters
    ----------
    :param params:
    :param gparams:

    :type lr: Theano Variable
    :param lr: learning rate

    :type momentum: float >= 0.
    :param momentum: parameter updates momentum.

    :type decay: float >= 0.
    :param decay: learning rate decay over each update.

    :type nesterov: boolean.
    :param nesterov: Whether to apply Nesterov momentum.
    """
    updates = []

    # learning rate decay
    iterations = theano.shared(value=numpy.asarray(0., dtype=theano.config.floatX))
    lr *= (1.0 / (1.0 + decay * iterations))
    updates.append((iterations, iterations + 1.))

    for p, g in zip(params, gparams):
        m = theano.shared(value=numpy.zeros(p.get_value(borrow=True).shape,
                                            dtype=theano.config.floatX))  # momentum
        v = momentum * m - lr * g  # velocity
        updates.append((m, v))

        if nesterov:
            new_p = p + momentum * v - lr * g
        else:
            new_p = p + v

        updates.append((p, new_p))

    return updates


def rmsprop(params, gparams, lr=0.001, rho=0.9, epsilon=1e-6):
    """
    rmsprop optimizer

    Parameters
    ----------
    :param params:
    :param gparams:

    :type lr: Theano Variable
    :param lr: learning rate

    :type rho: always set to 0.9
    :param rho: update coefficient of accumulators

    :type epsilon: always set to 1e-6
    :param epsilon: for computation robustness
    """
    updates = []
    accumulators = [theano.shared(value=numpy.zeros(p.get_value(borrow=True).shape,
                                                    dtype=theano.config.floatX))
                    for p in params]
    for p, g, a in zip(params, gparams, accumulators):
        new_a = rho * a + (1 - rho) * (g ** 2)
        updates.append((a, new_a))

        new_p = p - lr * g / T.sqrt(new_a + epsilon)
        updates.append((p, new_p))

        return updates


def adagrad(params, gparams, lr=0.001, epsilon=1e-6):
    """
    adagrad optimizer

    Parameters
    ----------
    :param params:
    :param gparams:

    :type lr: Theano Variable
    :param lr: learning rate

    :type epsilon: always set to 1e-6
    :param epsilon: for computation robustness
    """
    updates = []
    accumulators = [theano.shared(value=numpy.zeros(p.get_value(borrow=True).shape,
                                                    dtype=theano.config.floatX))
                    for p in params]
    for p, g, a in zip(params, gparams, accumulators):
        new_a = a + (g ** 2)
        updates.append((a, new_a))

        new_p = p - lr * g / T.sqrt(new_a + epsilon)
        updates.append((p, new_p))

    return updates


def adadelta(params, gparams, lr=0.001, rho=0.9, epsilon=1e-6):
    """
    adadelta optimizer

    Parameters
    ----------
    :param params:
    :param gparams:

    :type lr: Theano Variable
    :param lr: learning rate

    :type rho: always set to 0.9
    :param rho: update coefficient of accumulators

    :type epsilon: always set to 1e-6
    :param epsilon: for computation robustness
    """
    updates = []
    accumulators = [theano.shared(value=numpy.zeros(p.get_value(borrow=True).shape,
                                                    dtype=theano.config.floatX))
                    for p in params]
    delta_accumulators = [theano.shared(value=numpy.zeros(p.get_value(borrow=True).shape,
                                                    dtype=theano.config.floatX))
                    for p in params]

    for p, g, a, d_a in zip(params, gparams, accumulators, delta_accumulators):
        # update accumulator
        new_a = rho * a + (1 - rho) * (g ** 2)
        updates.append((a, new_a))

        # use the new accumulator and the *old* delta_accumulator
        update = g * T.sqrt(d_a + epsilon) / T.sqrt(new_a + epsilon)
        new_p = p - lr * update
        updates.append((p, new_p))

        # Accumulate the delta_x
        new_d_a = rho * d_a + (1 - rho) * (update ** 2)
        updates.append((d_a, new_d_a))

    return updates


def adam(params, gparams, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    """
    adam optimizer

    Parameters
    ----------
    :param params:
    :param gparams:

    :type lr: Theano Variable
    :param lr: learning rate

    :type beta_1: always set to 0.9
    :param beta_1: coefficient for momentum

    :type beta_2: always set to 0.999
    :param beta_2: coefficient for velocity

    :type epsilon: always set to 1e-6
    :param epsilon: for computation robustness
    """
    updates = []

    # learning rate dynamically determine
    iterations = theano.shared(value=numpy.asarray(0., dtype=theano.config.floatX))
    updates.append((iterations, iterations + 1.))

    t = iterations + 1
    lr_t = lr * T.sqrt(1 - T.pow(beta_2, t)) / (1 - T.pow(beta_1, t))

    for p, g in zip(params, gparams):
        # zero init of moment
        m = theano.shared(value=numpy.zeros(p.get_value(borrow=True).shape,
                                            dtype=theano.config.floatX))
        # zero init of velocity
        v = theano.shared(value=numpy.zeros(p.get_value(borrow=True).shape,
                                            dtype=theano.config.floatX))

        m_t = (beta_1 * m) + (1 - beta_1) * g
        v_t = (beta_2 * v) + (1 - beta_2) * (g ** 2)
        p_t = p - lr_t * m_t / (T.sqrt(v_t) + epsilon)

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))

    return updates









