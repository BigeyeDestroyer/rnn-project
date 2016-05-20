from ntm_cell import *
from optimizers.optimizers import *
import h5py


class NTM(object):
    def __init__(self, input_dim=8, output_dim=8, mem_size=128,
                 mem_width=20, layer_sizes=[100], num_reads=1,
                 batch_size=16, num_writes=1, shift_width=3,
                 eps=1e-12, min_grad=-10, max_grad=10,
                 optimizer='adam'):
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
        self.min_grad = min_grad
        self.max_grad = max_grad
        self.optimizer = optimizer
        # The input sequences with size
        # (sequence_length, batch_size, input_dim)
        self.X = T.tensor3('X')
        # The output sequences with size
        # (sequence_length, batch_size, input_dim)
        self.Y = T.tensor3('Y')

        self.lr = T.scalar('lr')  # the learning rate

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
        self.train_test_funcs()

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

    def train_test_funcs(self):
        # The target outputs, with size
        # (sequence_length, batch_size, output_dim)

        cost = self.negative_log_likelihood(Y=self.Y)

        gparams = []
        for param in self.cell.params:
            gparam = T.clip(T.grad(cost=cost, wrt=param),
                            self.min_grad, self.max_grad)
            gparams.append(gparam)

        # eval(): string to function
        optimizer = eval(self.optimizer)
        updates = optimizer(self.cell.params, gparams, self.lr)

        self.train = theano.function(inputs=[self.X, self.Y, self.lr],
                                     outputs=cost, updates=updates)
        self.pred = theano.function(inputs=[self.X],
                                    outputs=self.outputs)

    def save_to_file(self, file_name, file_index=None):
        """
        This function stores the trained params to '*.h5' file

        Parameters
        ----------
        :type file_name: str
        :param file_name: the directory with name to store trained parameters

        :type file_index: str, generated as str(1)
        :param file_index: if parameters here are snapshot,
                           then we need to add index to file name
        """
        if file_index is not None:
            file_name = file_name[:-3] + str(file_index) + '.h5'

        f = h5py.File(file_name)
        for p in self.cell.params:
            f[p.name] = p.get_value()
        f.close()











