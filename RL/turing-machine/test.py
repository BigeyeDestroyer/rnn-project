import theano
import theano.tensor as T
import numpy
from head import Head

model = Head()
print model.b_add.get_value().shape
print type(model.b_add.get_value())
