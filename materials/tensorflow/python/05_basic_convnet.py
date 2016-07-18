import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

def weight_variable(shape):
    """
    Helper functions to create a weight variable
    initialized with a normal distribution

    Parameters
    ----------
    shape : list
            Size of weight variable
    """
    initial = tf.random_normal(shape=shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    '''
    Helper function to create a bias variable
    initialized with a constant value.

    Parameters
    ----------
    shape : list
            Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)

def corrupt(x):
    """
    Take an input tensor and add uniform masking.

    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.

    Returns
    -------
    x_corrupted : Tensor
                  50 pct of values corrupted.
    """
    return tf.mul(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))

def montage(W):
    """
    Draws all filters (n_input * n_output filters) as a
    montage image separated by 1 pixel borders.

    Parameters
    ----------
    W : Tensor
        Input tensor to create montage of.
        With size (height, width, n_input, n_output)

    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    W = np.reshape(W, [W.shape[0], W.shape[1], 1, W.shape[2] * W.shape[3]])
    n_plots = int(np.ceil(np.sqrt(W.shape[-1])))
    m = np.ones(
        (W.shape[0] * n_plots + n_plots + 1,
         W.shape[1] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < W.shape[-1]:
                m[1 + i + i * W.shape[0]: 1 + i + (i + 1) * W.shape[0],
                  1 + j + j * W.shape[1]: 1 + j + (j + 1) * W.shape[1]] = (
                    np.squeeze(W[:, :, :, this_filter]))
    return m

def montage_batch(images):
    """
    Draws all images as a montage image
    separated by 1 pixel borders.

    Parameters
    ----------
    batch : Tensor
            Input tensor to create montage of.
            With size (batch_size, height, width, channels)

    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    img_h = images.shape[1]
    img_w = images.shape[2]

    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    m = np.ones(
        (images.shape[1] * n_plots + n_plots + 1,
         images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter, ...]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w, :] = this_img
    return m

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Since x is currently [batch, height * width], we need to reshape
# to a 4-D tensor to use it in a convolutional graph.
# If one component of `shape` is the special value -1, the size of
# that dimension is computed so that the total size remains constant.
# Since we haven't defined the batch dimension's shape yet, we use
# -1 to denote this dimension should not change size.
x_tensor = tf.reshape(x, shape=[-1, 28, 28, 1])

# Setup the first convolutional layer, weight matrix is
# [height, width, input_channel, output_channel]
# %% We'll setup the first convolutional layer
# Weight matrix is [height x width x input_channels x output_channels]
filter_size = 5
n_filters_1 = 16
W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])

# Bias is with size
# [output_channels]
b_conv1 = bias_variable([n_filters_1])

# Build a graph which does the first layer of convolution:
# We define our stride as batch x height x width x channels
# instead of pooling, we use strides of 2 and more layers
# with smaller filters.
#
# padding = 'SAME' means:
# out_height = ceil(float(in_height) / float(strides[1]))
# out_width = ceil(float(in_width) / float(strides[2]))
#
# padding = 'VALID' means:
# out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
# out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))

h_conv1 = tf.nn.relu(
    tf.nn.conv2d(input=x_tensor,
                 filter=W_conv1,
                 strides=[1, 2, 2, 1],
                 padding='SAME') +
    b_conv1)

n_filters_2 = 16
W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = bias_variable([n_filters_2])
h_conv2 = tf.nn.relu(
    tf.nn.conv2d(input=h_conv1,
                 filter=W_conv2,
                 strides=[1, 2, 2, 1],
                 padding='SAME') +
    b_conv2)

# Reshape so we can connect to a fully-connected layer:
h_conv2_flat = tf.reshape(h_conv2, [-1, 7 * 7 * n_filters_2])

# Create a fully-connected layer:
n_fc = 1024
W_fc1 = weight_variable([7 * 7 * n_filters_2, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

# Add dropout for regularizing and to reduce overfitting
#
# keep_prob:
# outputs the input element scaled up by 1 / keep_prob,
# otherwise outputs 0
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Add the final softmax layer
W_fc2 = weight_variable([n_fc, 10])
b_fc2 = bias_variable([10])
y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define loss/eval/training functions
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

# Monitor accuracy
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# Create a new session to actually
# perform the initialization of
# the variables
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Train in minibatches and report accuracy:
batch_size = 100
n_epochs = 5
for epoch_i in range(n_epochs):
    for batch_i in range(mnist.train.num_examples // batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})

    print(sess.run(accuracy,
                   feed_dict={x: mnist.validation.images,
                              y: mnist.validation.labels,
                              keep_prob: 1.0}))