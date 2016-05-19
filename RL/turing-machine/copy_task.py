import numpy

""" From tensor flow
length = 20
bits = 6
seq = numpy.zeros([length, bits + 2], dtype=numpy.float32)
for idx in xrange(length):
    seq[idx, 2: bits + 2] = numpy.random.rand(bits).round()

output = list(seq)
"""

input_size = 4
sequence_length = 6
batch_size = 16


def generate_copy_sequences(input_size, sequence_length, batch_size):
    """
    :type input_size: int
    :param input_size: input dim, where the last dim is for special token

    :type sequence_length: int
    :param sequence_length: length of sequence

    :type batch_size: int
    :param batch_size: to dynamically change the batch size

    :return:
    :type input_sequences: ndarray with size
                           (2 * sequence_length + 1, batch_size, input_size)
    param input_sequences: first half the generated random sequence
                           last half 0's

    :type output_sequences: ndarray with size
                           (2 * sequence_length + 1, batch_size, input_size)
    :param output_sequences: first half 0's
                             last half the generated random sequence
    """
    batch_input = []
    batch_output = []
    for idx in range(batch_size):
        # last dim of input_size is for dilemer
        sequence = numpy.random.binomial(
            1, 0.5, (sequence_length, input_size - 1)).astype(numpy.uint8)

        input_sequence = numpy.zeros(
            (sequence_length * 2 + 1, input_size), dtype=numpy.float32)
        output_sequence = numpy.zeros(
            (sequence_length * 2 + 1, input_size), dtype=numpy.float32)

        input_sequence[: sequence_length, :-1] = sequence
        input_sequence[sequence_length, -1] = 1
        output_sequence[sequence_length + 1:, :-1] = sequence

        batch_input.append(input_sequence[:, numpy.newaxis, :])
        batch_output.append(output_sequence[:, numpy.newaxis, :])

    return [numpy.concatenate(tuple(batch_input), axis=1),
            numpy.concatenate(tuple(batch_output), axis=1)]

sequence_input, \
sequence_output = generate_copy_sequences(
    input_size, sequence_length, batch_size)

print sequence_input.shape

print sequence_output.shape

