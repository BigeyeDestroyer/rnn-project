import numpy


def generate_copy_sequences(input_size_orig, sequence_length, batch_size):
    """
    :type input_size_orig: int
    :param input_size_orig: input dim, where the last dim is for special token

    :type sequence_length: int
    :param sequence_length: length of sequence

    :type batch_size: int
    :param batch_size: to dynamically change the batch size

    :return:
    :type input_sequences: ndarray with size
                           (2 * sequence_length + 2, batch_size, input_size + 2)
    param input_sequences: first half the generated random sequence
                           last half 0's

    :type output_sequences: ndarray with size
                           (2 * sequence_length + 2, batch_size, input_size + 2)
    :param output_sequences: first half 0's
                             last half the generated random sequence
    """
    input_size = input_size_orig - 2
    batch_input = []
    batch_output = []
    for idx in range(batch_size):
        # last two dims of input are for start and end symbol
        sequence = numpy.random.binomial(
            1, 0.5, (sequence_length, input_size)).astype(dtype=numpy.uint8)

        input_sequence = numpy.zeros(
            (2 * sequence_length + 2, input_size + 2), dtype=numpy.float32)
        output_sequence = numpy.zeros(
            (2 * sequence_length + 2, input_size + 2), dtype=numpy.float32)

        input_sequence[0, -2] = 1  # the start symbol
        input_sequence[1: sequence_length + 1, :-2] = sequence
        input_sequence[sequence_length + 1, -1] = 1  # the end symbol
        output_sequence[sequence_length + 2:, :-2] = sequence

        batch_input.append(input_sequence[:, numpy.newaxis, :])
        batch_output.append(output_sequence[:, numpy.newaxis, :])

    return [numpy.concatenate(tuple(batch_input), axis=1),
            numpy.concatenate(tuple(batch_output), axis=1)]

