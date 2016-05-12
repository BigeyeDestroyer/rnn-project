__author__ = 'ray'
import load_data
import theano.tensor as T
import numpy
import MLP
import theano
import os
import timeit
import sys

learning_rate = 0.01
L1_reg = 0.00
L2_reg = 0.0001
n_epochs = 1000
dataset = 'mnist.pkl.gz'
batch_size = 20
n_hidden = 500

"""
Demonstrate stochastic gradient descent optimization
for a multi-layer perceptron

:type learning_rate: float
:param learning_rate: learning rate used for the stochastic method

:type L1_reg: float
:param L1_reg: L1-norm's weight when added to the cost

:type L2_reg: float
:param L2_reg: L2-norm's weight when added to the cost

:type n_epochs: int
:param n_epochs: maximal number of epochs to run the optimizer

:type dataset: string
:param dataset: the path of the dataset
"""

datasets = load_data.load_data(dataset)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

# compute number of minibatches
n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

###################
# BUILD THE MODEL #
###################
print '... building the model'

# allocate the symbolic variables for the data
index = T.lscalar() # index to a [mini]batch
x = T.matrix('x') # the data
y = T.ivector('y') # the labels

rng = numpy.random.RandomState(1234)

# construct the MLP class
classifier = MLP.MLP(
    rng=rng,
    input=x,
    n_in=28 * 28,
    n_hidden=n_hidden,
    n_out=10,
    y=y
)

# the cost we minimize during the training
cost = (
    classifier.negative_log_likelihood
    + L1_reg * classifier.L1
    + L2_reg * classifier.L2_sqr
)

# compiling the testing and validation model
test_model = theano.function(
    inputs=[index],
    outputs=classifier.errors,
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    inputs=[index],
    outputs=classifier.errors,
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

# compute the gradients
gparams = [T.grad(cost, param) for param in classifier.params]

# specify how to update the parameters
updates = [
    (param, param - learning_rate * gparam)
    for param, gparam in zip(classifier.params, gparams)
]

# compile the training model
train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

###############
# TRAIN MODEL #
###############
print '... training'

# early stopping parameters
patience = 1000 #
patience_increase = 2 # wait this much longer when a
                      # new best if found
improvement_threshold = 0.995 # a relative improvement of
                              # this much is considered significant
validation_frequency = min(n_train_batches, patience / 2) # number of batches for validation

best_validation_loss = numpy.inf
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):
        minibatch_avg_cost = train_model(minibatch_index)
        # iteration number, for the minibatch
        iter = (epoch - 1) * n_train_batches + minibatch_index
        if (iter + 1) % validation_frequency == 0:
            validation_losses = [validate_model(i)
                                 for i in xrange(n_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)
            print(
                'epoch %i, minibatch %i/%i, validation error %f %%' %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100
                )
            )

            # if we got the best validation score until now
            if(this_validation_loss < best_validation_loss):
                #improve patience if loss improvement is good enough
                if(this_validation_loss < best_validation_loss * improvement_threshold):
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_losses = [test_model(i)
                               for i in xrange(n_test_batches)]
                test_score = numpy.mean(test_losses)
                print(
                    ('    epoch %i, minibatch %i/%i, test error of '
                     'best model %f %%') %
                    (epoch, minibatch_index + 1, n_train_batches,
                     test_score * 100.)
                )

        if patience <= iter:
            done_looping = True
            break

end_time = timeit.default_timer()

print(
    ('Optimization complete. Best validation score of %f %%'
     'obtained at iteration %i, with test performance %f %%') %
    (best_validation_loss * 100., best_iter + 1, test_score * 100.)
)

print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                      ' ran for %.2fm' % ((end_time - start_time) / 60.))



