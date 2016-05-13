import numpy

numpy_rng = numpy.random.RandomState(1234)

a = numpy_rng.normal(size=(2, 2))
print a
b = numpy_rng.normal(size=(2, 2))
print b
