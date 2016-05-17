from common.utils import *
import theano
import theano.tensor as T
import numpy

mem_size = 128
mem_width = 20




def cosine_sim(k, M):
    """
    :type k: tensor variable with size (batch_size, mem_width)
    :param k: input to calculate similarity

    :type M: tensor variable with size (mem_size, mem_width)
    :param M: memory matrix

    :return: similarity measure with size (batch_size, mem_size)
    """
    k_lengths = T.sqrt(T.sum(k ** 2, axis=1)).dimshuffle((0, 'x'))
    k_unit = k / (k_lengths + 1e-5)

    M_lengths = T.sqrt(T.sum(M ** 2, axis=1)).dimshuffle((0, 'x'))
    M_unit = M / (M_lengths + 1e-5)
    return T.dot(k_unit, T.transpose(M_unit))

k = T.matrix('k')  # with size (batch_size, mem_width)
M = T.matrix('M')  # with size (mem_size, mem_width)
cosine_sim = cosine_sim(k, M)

fn_unit = theano.function(inputs=[k, M], outputs=cosine_sim)

k_in = numpy.random.randn(5, mem_width)
M_in = numpy.random.randn(mem_size, mem_width)

cos_sim = fn_unit(k_in, M_in)




