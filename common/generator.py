from .ops import *

def block_cbn(x, labels, out_channels, num_classes, name):
    with tf.variable_scope(name):
        bn0 = ConditionalBatchNorm(num_classes, name='cbn_0')
        bn1 = ConditionalBatchNorm(num_classes, name='cbn_1')

        x_0 = x
        x = tf.nn.relu(bn0(x, labels))
        x = usample(x)
        x = conv2d(x, out_channels, ksize = 3, stride = 1, name='conv1')
        x = tf.nn.relu(bn1(x, labels))
        x = conv2d(x, out_channels, ksize = 3, stride = 1, name='conv2')

        x_0 = usample(x_0)
        x_0 = conv2d(x_0, out_channels, ksize = 1, stride = 1, name='conv3')
    return x_0 + x

def block_bn(x, out_channels, name):
    with tf.variable_scope(name):
        x_0 = x
        x = tf.nn.relu(batch_norm(x, name='bn0'))
        x = usample(x)
        x = conv2d(x, out_channels, ksize=3, stride=1, name='conv1')
        x = tf.nn.relu(batch_norm(x, name='bn1'))
        x = conv2d(x, out_channels, ksize=3, stride=1, name='conv2')

        x_0 = usample(x_0)
        x_0 = conv2d(x_0, out_channels, ksize=1, stride=1, name='conv3')
    return x_0 + x

def generator_uncond_cifar10(num_sample, cfg, z=None):
    c = [4, 4, 4]
    iFilterDimsG = cfg.iFilterDimsG

    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):

        if z is None:
            h0 = tf.random_normal(shape=[num_sample, cfg.iDimsZ])
        else:
            h0 = z

        h0 = linear(h0, 4 * 4 * (iFilterDimsG * 4))
        h0 = tf.reshape(h0, [-1, 4, 4, iFilterDimsG * 4])
        for i in range(3):
            h0 = block_bn(h0, iFilterDimsG * c[i], 'block{}'.format(i))

        h0 = tf.nn.relu(batch_norm(h0, name='bn_last'))
        h0 = conv2d(h0, 3, ksize=3, stride=1, name='conv_last')
        h0 = tf.nn.tanh(h0)

        return h0


def generator_uncond_stl10(num_sample, cfg, z=None):
    c = [4, 4, 4]
    iFilterDimsG = cfg.iFilterDimsG

    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):

        if z is None:
            h0 = tf.random_normal(shape=[num_sample, cfg.iDimsZ])
        else:
            h0 = z

        h0 = linear(h0, 6 * 6 * (iFilterDimsG * 4))
        h0 = tf.reshape(h0, [-1, 6, 6, iFilterDimsG * 4])

        for i in range(3):
            h0 = block_bn(h0, iFilterDimsG * c[i], 'block{}'.format(i))

        h0 = tf.nn.relu(batch_norm(h0, name='bn_last'))
        h0 = conv2d(h0, 3, ksize=3, stride=1, name='conv_last')
        h0 = tf.nn.tanh(h0)

        return h0

def generator_cond_cifar(num_sample, cfg, num_classes, labels, z=None):
    c = [4, 4, 4]
    iFilterDimsG = cfg.iFilterDimsG

    with tf.variable_scope('generator', tf.AUTO_REUSE):

        if z is None:
            h0 = tf.random_normal(shape=[num_sample, cfg.iDimsZ])
        else:
            h0 = z

        h0 = linear(h0, 4 * 4 * (iFilterDimsG * 4))  # linear 4x4
        h0 = tf.reshape(h0, [-1, 4, 4, iFilterDimsG * 4])  # reshape 4x4

        for i in range(3):
            h0 = block_cbn(h0, labels, iFilterDimsG * c[i], num_classes, 'block{}'.format(i))

        h0 = tf.nn.relu(batch_norm(h0))
        h0 = conv2d(h0, 3, ksize=3, stride=1, name='conv_last')
        h0 = tf.nn.tanh(h0)

        return h0

def generator_cond_tiny(num_sample, cfg, num_classes, labels, z=None):
    c = [4, 4, 4, 4]
    iFilterDimsG = cfg.iFilterDimsG

    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):

        if z is None:
            h0 = tf.random_normal(shape=[num_sample, cfg.iDimsZ])
        else:
            h0 = z

        h0 = linear(h0, 4 * 4 * (iFilterDimsG * 4))  # linear 4x4
        h0 = tf.reshape(h0, [-1, 4, 4, iFilterDimsG * 4])  # reshape 4x4

        for i in range(4):
            h0 = block_cbn(h0, labels, iFilterDimsG * c[i], num_classes, 'block{}'.format(i))

        h0 = tf.nn.relu(batch_norm(h0, name='bn_last'))
        h0 = conv2d(h0, 3, ksize=3, stride=1, name='conv_last')
        h0 = tf.nn.tanh(h0)

        return h0
