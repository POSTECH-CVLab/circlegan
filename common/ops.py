import warnings
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import linalg_ops

__enable_bias__ = True

__enable_wn__ = True
__scale_weight_after_init__ = True

__init_layer_stddev__ = 1.0
__init_weight_stddev__ = 0.01 # small __init_weight_stddev__, such as 0.01, along with Adam, seems benefit the generalization.
__init_distribution_type__ = 'uniform' # truncated_normal, normal, uniform, orthogonal

def set_enable_bias(boolvalue):
    global __enable_bias__
    __enable_bias__ = boolvalue


def set_enable_wn(boolvalue):
    global __enable_wn__
    __enable_wn__ = boolvalue


def set_init_weight_stddev(stddev):
    global __init_weight_stddev__
    __init_weight_stddev__ = stddev


def set_init_layer_stddev(stddev):
    global __init_layer_stddev__
    __init_layer_stddev__ = stddev


def set_scale_weight_after_init(boolvalue):
    global __scale_weight_after_init__
    __scale_weight_after_init__ = boolvalue


def identity(inputs):
    return inputs


def _l2normalize(v, eps=1e-12):
  """l2 normize the input vector."""
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def batch_norm(input, epsilon=0.001, name='batchnorm'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        input_shape = input.get_shape().as_list()

        if len(input_shape) == 4:
            h_axis, w_axis, c_axis = [1, 2, 3]
            reduce_axis = [0, h_axis, w_axis]
            params_shape = [1 for i in range(len(input_shape))]
            params_shape[c_axis] = input_shape[c_axis]

            offset = tf.get_variable('offset', shape=params_shape, initializer=tf.zeros_initializer())
            scale = tf.get_variable('scale', shape=params_shape, initializer=tf.ones_initializer())

            batch_mean, batch_variance = tf.nn.moments(input, reduce_axis, keep_dims=True)
            #x_batch = (input - batch_mean) / (tf.sqrt(batch_variance + eps))
            #outputs = x_batch * scale + offset
            outputs = tf.nn.batch_normalization(input, batch_mean, batch_variance, offset, scale, epsilon)

    return outputs

class ConditionalBatchNorm(object):
    """Conditional BatchNorm.

    For each  class, it has a specific gamma and beta as normalization variable.
    """

    def __init__(self, num_categories, epsilon=0.001, name='conditional_batch_norm'):
        with tf.variable_scope(name):
            self.name = name
            self.num_categories = num_categories
            self.epsilon = epsilon

    def __call__(self, inputs, labels, srm=False):
        inputs = tf.convert_to_tensor(inputs)
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[3:4]
        axis = [0, 1, 2]
        shape = tf.TensorShape([self.num_categories]).concatenate(params_shape)

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.beta = tf.get_variable(
                'beta', shape,
                initializer=tf.zeros_initializer())
            self.gamma = tf.get_variable(
                'gamma', shape,
                initializer=tf.ones_initializer())
            beta = tf.gather(self.beta, labels)
            beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)
            gamma = tf.gather(self.gamma, labels)
            gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
            mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)
            outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, self.epsilon)
            return outputs

def conv2d(input, output_dim, ksize=3, stride=1, padding='SAME', bBias=False, name='conv2d'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        input_shape = input.get_shape().as_list()
        h_axis, w_axis, c_axis = [1, 2, 3]
        strides = [1, stride, stride, 1]
        if __enable_wn__:
            w = tf.get_variable('w', [ksize, ksize, input_shape[c_axis], output_dim], initializer=initializer(__init_distribution_type__, [0, 1, 2], __init_weight_stddev__))
            g = tf.get_variable('g', initializer=tf.ones_like(tf.reduce_sum(w, [0, 1, 2], keep_dims=True)))
            w = g * tf.nn.l2_normalize(w, [0, 1, 2])
        elif __scale_weight_after_init__:
            scale = __init_layer_stddev__ / tf.sqrt(float(ksize * ksize * input_shape[c_axis])) / __init_weight_stddev__
            w = tf.get_variable('w', [ksize, ksize, input_shape[c_axis], output_dim], initializer=initializer(__init_distribution_type__, [0, 1, 2], __init_weight_stddev__)) * scale
        else:
            scale = __init_layer_stddev__ / tf.sqrt(float(ksize * ksize * input_shape[c_axis])) / __init_weight_stddev__
            w = tf.get_variable('w', [ksize, ksize, input_shape[c_axis], output_dim], initializer=initializer(__init_distribution_type__, [0, 1, 2], __init_weight_stddev__ * scale))

        x = tf.nn.conv2d(input, w, strides=strides, padding=padding)

        if __enable_bias__ or bBias:
            b = tf.get_variable('b', initializer=tf.constant_initializer(0.0), shape=[output_dim])
            x = tf.nn.bias_add(x, b)

    return x

def linear(input, output_size, bBias=False, name='linear'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        if len(input.get_shape().as_list()) > 2:
            warnings.warn('using ops \'linear\' with input shape' + str(input.get_shape().as_list()))
            input = tf.reshape(input, [input.get_shape().as_list()[0], -1])

        if __enable_wn__:
            w = tf.get_variable('w', [input.get_shape().as_list()[1], output_size], initializer=initializer(__init_distribution_type__, [0], __init_weight_stddev__))
            g = tf.get_variable('g', initializer=tf.ones_like(tf.reduce_sum(tf.square(w), [0], keep_dims=True)))
            w = g * tf.nn.l2_normalize(w, [0])
        elif __scale_weight_after_init__:
            scale = __init_layer_stddev__ / tf.sqrt(float(input.get_shape().as_list()[1])) / __init_weight_stddev__
            w = tf.get_variable('w', [input.get_shape().as_list()[1], output_size], initializer=initializer(__init_distribution_type__, [0], __init_weight_stddev__)) * scale
        else:
            scale = __init_layer_stddev__ / tf.sqrt(float(input.get_shape().as_list()[1])) / __init_weight_stddev__
            w = tf.get_variable('w', [input.get_shape().as_list()[1], output_size], initializer=initializer(__init_distribution_type__, [0], __init_weight_stddev__ * scale))


        x = tf.matmul(input, w)
        if __enable_bias__ or bBias:
            b = tf.get_variable('b', initializer=tf.constant_initializer(0.0), shape=[output_size])
            x = tf.nn.bias_add(x, b)

    return x

def image_nn_double_size(input, name='resize'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        h_axis, w_axis, c_axis = [1, 2, 3]
        input = tf.concat([input, input, input, input], axis=c_axis)
        input = tf.depth_to_space(input, 2)

    return input

def im_resize (input, shape, name='bilinear_resize'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        input = tf.image.resize_bilinear(input, shape)

    return input


def dsample(input, name='dsample'):
  """Downsamples the image by a factor of 2."""

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      input = tf.nn.avg_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
  return input

def usample(input, name='nn_resize'):
    _, nh, nw, nx = input.get_shape().as_list()
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        input = tf.image.resize_nearest_neighbor(input, [nh * 2, nw * 2])

    return input

def noise(input, stddev, bAdd=False, bMulti=True, keep_prob=None, name='noise'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        if bAdd:
            input = input + tf.truncated_normal(tf.shape(input), 0, stddev, name=name)

        if bMulti:
            if keep_prob is not None:
                stddev = tf.sqrt((1 - keep_prob) / keep_prob)  # get 'equivalent' stddev to dropout of keep_prob
            input = input * tf.truncated_normal(tf.shape(input), 1, stddev, name=name)

    return input


def dropout(input, drop_prob, name='dropout'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        keep_prob = 1.0 - drop_prob
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(tf.shape(input))
        binary_tensor = tf.floor(random_tensor)
        input = input * binary_tensor * tf.sqrt(1.0 / keep_prob)

    return input


def activate(input, oAct):

    with tf.variable_scope(oAct):

        if oAct == 'elu':
            input = tf.nn.elu(input)

        elif oAct == 'relu':
            input = tf.nn.relu(input)

        elif oAct == 'lrelu':
            input = tf.nn.leaky_relu(input)

        elif oAct == 'softmax':
            input = tf.nn.softmax(input)

        elif oAct == 'tanh':
            input = tf.nn.tanh(input)

        elif oAct == 'crelu':
            input = tf.nn.crelu(input)

        elif oAct == 'selu':
            input = tf.nn.selu(input)

        elif oAct == 'swish':
            input = tf.nn.sigmoid(input) * input

        elif oAct == 'softplus':
            input = tf.nn.softplus(input)

        elif oAct == 'softsign':
            input = tf.nn.softsign(input)

        else:
            assert oAct == 'none'

    return input


def lnoise(input, fNoise, fDrop):

    if fNoise > 0:
        input = noise(input=input, stddev=fNoise, bMulti=True, bAdd=False)

    if fDrop > 0:
        input = dropout(input=input, drop_prob=fDrop)

    return input


def minibatch_feature(input, n_kernels=100, dim_per_kernel=5, name='minibatch'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        if len(input.get_shape()) > 2:
            input = tf.reshape(input, [input.get_shape().as_list()[0], -1])

        batchsize = input.get_shape().as_list()[0]

        x = linear(input, n_kernels * dim_per_kernel)
        x = tf.reshape(x, [-1, n_kernels, dim_per_kernel])

        mask = np.zeros([batchsize, batchsize])
        mask += np.eye(batchsize)
        mask = np.expand_dims(mask, 1)
        mask = 1. - mask
        rscale = 1.0 / np.sum(mask)

        abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(x, 3) - tf.expand_dims(tf.transpose(x, [1, 2, 0]), 0)), 2)
        masked = tf.exp(-abs_dif) * mask
        dist = tf.reduce_sum(masked, 2) * rscale

    return dist


def channel_concat(x, y):

    x_shapes = x.get_shape().as_list()
    y_shapes = y.get_shape().as_list()
    assert y_shapes[0] == x_shapes[0]

    y = tf.reshape(y, [y_shapes[0], 1, 1, y_shapes[1]]) * tf.ones([y_shapes[0], x_shapes[1], x_shapes[2], y_shapes[1]])
    return tf.concat([x, y], 3)


def normalized_orthogonal_initializer(flatten_axis, stddev=1.0):

    def _initializer(shape, dtype=None, partition_info=None):

        if len(shape) < 2:
            raise ValueError("The tensor to initialize must be at least two-dimensional")

        num_rows = 1
        for dim in [shape[i] for i in flatten_axis]:
            num_rows *= dim
        num_cols = shape[list(set(range(len(shape))) - set(flatten_axis))[0]]

        flat_shape = (num_cols, num_rows) if num_rows < num_cols else (num_rows, num_cols)

        a = random(flat_shape, type='uniform', stddev=1.0)
        q, r = linalg_ops.qr(a, full_matrices=False)

        q *= np.sqrt(flat_shape[0])

        if num_rows < num_cols:
            q = tf.matrix_transpose(q)

        return stddev * tf.reshape(q, shape)

    return _initializer


def initializer(type, flatten_axis, stddev):

    if type == 'normal':
        return tf.random_normal_initializer(stddev=stddev)

    elif type == 'uniform':
        return tf.random_uniform_initializer(minval=-stddev * tf.sqrt(3.0), maxval=stddev * tf.sqrt(3.0))

    elif type == 'truncated_normal':
        return tf.truncated_normal_initializer(stddev=stddev * tf.sqrt(1.3))

    elif type == 'orthogonal':
        return normalized_orthogonal_initializer(flatten_axis, stddev=stddev)


def random(shape, type, stddev):

    if type == 'normal':
        return tf.random_normal(shape, stddev=stddev)

    elif type == 'uniform':
        return tf.random_uniform(shape, minval=-stddev * tf.sqrt(3.0), maxval=stddev * tf.sqrt(3.0))

    elif type == 'truncated_normal':
        return tf.truncated_normal(shape, stddev=stddev * tf.sqrt(1.3))
