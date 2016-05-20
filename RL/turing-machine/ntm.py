from copy_task import *
from ntm_cell import *
import theano


class NTM(object):
    def __init__(self, input_dim=8, output_dim=8, mem_size=128,
                 mem_width=20, layer_sizes=[100], num_reads=1,
                 batch_size=16, num_writes=1, shift_width=3,
                 eps=1e-12):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mem_size = mem_size
        self.mem_width = mem_width
        self.layer_sizes = layer_sizes
        self.num_reads = num_reads
        self.batch_size = batch_size
        self.num_writes = num_writes
        self.shift_width = shift_width
        self.eps = eps
        # The input sequences with size
        # (sequence_length, batch_size, input_dim)
        self.X = T.tensor3('X')

        # Build the structure
        self.cell = NTMCell(input_dim=input_dim, output_dim=output_dim,
                            mem_size=mem_size, mem_width=mem_width,
                            layer_sizes=layer_sizes, num_reads=num_reads,
                            batch_size=batch_size, num_writes=num_writes,
                            shift_width=shift_width, eps=eps)

        # Compute the outputs along time
        # outputs is a tensor variable
        # with size (sequence_length, batch_size, output_dim)
        outputs, _ = theano.scan(fn=self.cell.step,
                                 sequences=self.X)
        self.outputs = outputs

    def negative_log_likelihood(self, Y):
        """
        :type Y: tensor variable with size
                 (sequence_length, batch_size, output_dim)
        :param Y: target outputs
        :return:
        """
        # reshape outputs => p_y
        # reshape Y => y
        p_y = T.reshape(self.outputs, (self.outputs.shape[0] * self.outputs.shape[1],
                                       self.outputs.shape[2]))
        p_y = T.clip(p_y, self.eps, 1 - self.eps)

        y = T.reshape(Y, (Y.shape[0] * Y.shape[1],
                          Y.shape[2]))

        # cost is with size
        # (sequence_length * batch_size, output_dim)
        cost = -T.mean(T.log(p_y) * y)

        return cost




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

model = NTM()
Y = T.tensor3('Y')

sequence_length = 20
input_sequences, output_sequences = \
    generate_copy_sequences(input_size_orig=input_dim,
                            sequence_length=sequence_length,
                            batch_size=batch_size)

fn_test = theano.function(inputs=[model.X, Y],
                          outputs=model.negative_log_likelihood(Y))


out_data = fn_test(input_sequences, output_sequences)

print input_sequences.shape
print output_sequences.shape
print type(out_data)
print out_data




