from .ops import *

def optimized_block(x, out_channels, name, act=tf.nn.relu):
    with tf.variable_scope(name):
        x_0 = x
        x = conv2d(x, out_channels, ksize = 3, stride = 1, name='conv1')
        x = act(x)
        x = conv2d(x, out_channels, ksize = 3, stride = 1, name='conv2')

        x = dsample(x)
        x_0 = dsample(x_0)
        x_0 = conv2d(x_0, out_channels, ksize = 1, stride = 1, name='conv3')
    return x + x_0

def block(x, out_channels, name, downsample=True, act=tf.nn.relu):
    with tf.variable_scope(name):
        input_channels = x.shape.as_list()[1]
        x_0 = x

        x = act(batch_norm(x, name = 'bn1'))
        x = conv2d(x, out_channels, ksize = 3, stride = 1, name='conv1')
        x = act(batch_norm(x, name = 'bn2'))
        x = conv2d(x, out_channels, ksize = 3, stride = 1, name='conv2')

    if downsample:
        x = dsample(x)
    if downsample or input_channels != out_channels:
        x_0 = conv2d(x_0, out_channels, ksize = 1, stride = 1, name='conv3')
        if downsample:
            x_0 = dsample(x_0)

    return x + x_0

def discriminator_uncond_cifar10(input, cfg):

    iFilterDimsD = cfg.iFilterDimsD
    c = [2, 2, 2]

    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

        h0 = input
        h0 = optimized_block(h0, iFilterDimsD * 2, 'optimized_block1_nosn')
        h0 = dropout(h0, cfg.fDropRate)

        hc = h0
        for i in range(3):
            hc = block(hc, iFilterDimsD * c[i], name="block{}".format(i), downsample=i<2)  # 32 * 32
            hc = dropout(hc, cfg.fDropRate)

        hc = tf.reduce_mean(hc, [1, 2])
        fc = tf.contrib.layers.flatten(hc)

        rf = tf.get_variable(
            name='rf',
            shape=[cfg.iK, iFilterDimsD * c[2]],
            initializer=tf.orthogonal_initializer())

        return fc, rf

def discriminator_uncond_stl10(input, cfg):

    iFilterDimsD = cfg.iFilterDimsD
    c = [4, 4, 4]

    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):

        h0 = input

        h0 = optimized_block(h0, iFilterDimsD * 4, 'optimized_block1_nosn')
        h0 = dropout(h0, cfg.fDropRate)

        hc = h0
        for i in range(3):
            hc = block(hc, iFilterDimsD * c[i], name="block{}".format(i), downsample=i<2)  # 48 * 48
            hc = dropout(hc, cfg.fDropRate)

        hc = tf.reduce_mean(hc, [1, 2])
        fc = tf.contrib.layers.flatten(hc)

        rf = tf.get_variable(
            name='rf',
            shape=[cfg.iK, iFilterDimsD * c[2]],
            initializer=tf.orthogonal_initializer())

        return fc, rf
