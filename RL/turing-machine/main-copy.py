import time
from layers.ntm import *
from copy_task import *
import os

""" params for model
"""
input_dim = 8
output_dim = 8
mem_size = 128
mem_width = 20
layer_sizes = [100]
num_reads = 1
batch_size = 16
num_writes = 1
shift_width = 3
eps = 1e-12
min_grad = -10
max_grad = 10
optimizer = 'adam'

""" params for training
"""
lr = 1e-3

dispFreq = 5  # print each 10 mini-batch
saveFreq = 100  # save the model every 100 iters
saveto = '/home/trunk/disk4/lurui/models/turing-machine'

max_iters = 10000
seq_length_max = 20
seq_length_min = 1

# suppose the beginning patience if 10% of the total epochs
patience = max_iters

""" build the model
"""
print('Building the model ...')
model = NTM(input_dim=input_dim, output_dim=output_dim,
            mem_size=mem_size, mem_width=mem_width,
            layer_sizes=layer_sizes, num_reads=num_reads,
            batch_size=batch_size, num_writes=num_writes,
            shift_width=shift_width, eps=eps, min_grad=min_grad,
            max_grad=max_grad, optimizer=optimizer)


""" Optimization
"""
print('Optimization ...')
history_cost = []  # cost for train, valid every validFreq

iters = 0  # number of mini-batches we've gone through
start_time = time.time()

try:
    while (iters < max_iters):
        iters += 1
        seq_length = numpy.random.randint(
            low=seq_length_min, high=seq_length_max + 1)
        input_batch, target_batch = generate_copy_sequences(
            input_size_orig=input_dim,
            sequence_length=seq_length,
            batch_size=batch_size)

        cost = model.train(input_batch, target_batch, lr)

        if numpy.mod(iters, dispFreq) == 0:
            print('Iters%d, Seq length: %d, Cost: %.6f' %
                  (iters, seq_length, cost))
            history_cost.append(cost)

        if numpy.mod(iters, saveFreq) == 0:
            # save model
            model_file = saveto + '/model_iter' + str(iters) + '.h5'
            if os.path.isfile(model_file):
                os.system('rm ' + model_file)
            print('Saving model at iter%d ...' % iters)
            model.save_to_file(model_file)

            # save cost
            cost_file = saveto + '/cost_turing.npz'
            numpy.savez(cost_file,
                        history_cost=history_cost)
            print('Done.')
except KeyboardInterrupt:
    print('Training interupted ...')
end_time = time.time()


# After interrupt, we save the model
# save the model
model_file = saveto + '/model_turing_final.h5'
if os.path.isfile(model_file):
    os.system('rm ' + model_file)
print('Saving model at iter%d ...' % iters)
model.save_to_file(model_file)

# save the costs
cost_file = saveto + '/cost_turing.npz'
numpy.savez(cost_file,
            history_cost=history_cost)
print('Done.')

print('The code run for %d iters, with %f min/iter' % (
    (iters + 1), (end_time - start_time) / 60. / (1. * (iters + 1))))
print('Training took %.1fs' % (end_time - start_time))
