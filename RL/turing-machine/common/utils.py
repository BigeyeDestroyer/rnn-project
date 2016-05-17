from collections import OrderedDict
import numpy
import theano


def floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)


def ortho_weight(ndim):
    """
    This function initializes the weights to be orthogonal vectors
    Parameters
    ----------
    :type ndim: int
    :param ndim: dimension of the input and hidden
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)


def init_weights(shape, name):
    """
    This function initializes weights to be shared variable
    with given 'shape' which is a list or tuple(mostly tuple),
    thus we need '*shape' to extract the ints in the tuple
    for convenience of function 'numpy.random.randn'
    Parameters
    ----------
    :type shape: tuple or list
    :param shape: shape of the weights to be generated
    :type name: string
    :param name: name of the weights
    """
    return theano.shared(value=floatX(numpy.random.randn(*shape) * 0.1),
                         name=name)


def glorot_uniform(shape, name):
    """
    This function initializes weights to be shared variable
    with given 'shape' which is a list or tuple(mostly tuple),
    Parameters
    ----------
    :type shape: tuple or list
    :param shape: shape of the weights to be generated
    :type name: string
    :param name: name of the weights
    """
    return theano.shared(
        value=numpy.asarray(
            numpy.random.uniform(
                low=-numpy.sqrt(6. / sum(shape)),
                high=numpy.sqrt(6. / sum(shape)),
                size=shape
            ),
            dtype=theano.config.floatX
        ),
        name=name)


def init_grads(shape, name):
    """
    This function initializes gradients of the model's weights
    Parameters
    ----------
    :type shape: tuple or list
    :param shape: shape of the grads to be generated,
                  the same with corresponding weights
    :type name: string
    :param name: name of the gradients
    """
    return theano.shared(value=floatX(numpy.zeros(shape)),
                         name=name)


def init_bias(size, name):
    """
    This function initializes bias of the model
    Parameters
    ----------
    :type size: int
    :param size: length of the bias
    :type name: string
    :param name: name of the bias
    """
    return theano.shared(value=floatX(numpy.zeros((size, ))),
                         name=name)


def init_array(array, name):
    """
    This function initializes an array
    with given values

    Parameters
    ----------
    :type array: 2D ndarray
    :param array: values to setup the shared variable

    :type name: string
    :param name: name of the shared variable
    """
    return theano.shared(
        value=numpy.asarray(
            array,
            dtype=theano.config.floatX
        ),
        name=name
    )


def init_scalar(value, name):
    """
    This function initializes a scalar
    Parameters
    ----------
    :type value: int
    :param value: length of the bias

    :type name: string
    :param name: name of the bias
    """
    return theano.shared(
        value=numpy.asarray(
            value,
            dtype=theano.config.floatX
        ),
        name=name
    )


def zipp(params, tparams):
    """ When we reload the model.
        Needed for the GPU stuff.
    :type tparams: Theano SharedVariable
    :param tparams: Model parameters
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """ When we pickle the model.
        Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def param_name(pp, name):
    return '%s_%s' % (pp, name)