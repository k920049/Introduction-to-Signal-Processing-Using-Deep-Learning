import tensorflow as tf
import numpy as np

# constants
X_dim = 256
Y_dim = 2
Z_dim = 256 * 256
value_lambda = 0.001

X = tf.placeholder(tf.float32, shape=[None, X_dim, X_dim, 1])
Y = tf.placeholder(tf.float32, shape=[None, Y_dim])
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

initializer = tf.contrib.layers.variance_scaling_initializer
activation_function = tf.nn.elu

custom_filter = np.ones(shape=[128, 256, 256, 1], dtype=np.float)
custom_filter[:, 255, :, :] = 0
custom_filter[:, :, 255, :] = 0

custom_filter = tf.constant(custom_filter, dtype=tf.float32)


def discriminator(x, name=None):
    with tf.name_scope(name, "discriminator", [x]) as scope:

        D_conv_1 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=[
                                    5, 5], padding='SAME', activation=activation_function)
        # [256, 256]
        D_mean_pool_1 = tf.nn.pool(D_conv_1, window_shape=[
                                   2, 2], pooling_type='AVG', padding='VALID', strides=[2, 2])
        # [128, 128]
        D_conv_2 = tf.layers.conv2d(D_mean_pool_1, filters=32, kernel_size=[
                                    3, 3], padding='SAME', activation=activation_function)
        # [128, 128]
        D_mean_pool_2 = tf.nn.pool(D_conv_2, window_shape=[
                                   2, 2], pooling_type='AVG', padding='VALID', strides=[2, 2])
        # [64, 64]
        D_conv_3 = tf.layers.conv2d(D_mean_pool_2, filters=64, kernel_size=[
                                    3, 3], padding='SAME', activation=activation_function)
        # [64, 64]
        D_mean_pool_3 = tf.nn.pool(D_conv_3, window_shape=[
                                   2, 2], pooling_type='AVG', padding='VALID', strides=[2, 2])
        # [32, 32]
        D_conv_4 = tf.layers.conv2d(D_mean_pool_3, filters=128, kernel_size=[
                                    3, 3], padding='SAME', activation=activation_function)
        # [32, 32]
        D_mean_pool_4 = tf.nn.pool(D_conv_4, window_shape=[
                                   2, 2], pooling_type='AVG', padding='VALID', strides=[2, 2])
        # [16, 16]
        D_conv_5 = tf.layers.conv2d(D_mean_pool_4, filters=256, kernel_size=[
                                    3, 3], padding='SAME', activation=activation_function)
        # [16, 16]
        D_mean_pool_5 = tf.nn.pool(D_conv_5, window_shape=[
                                   4, 4], pooling_type='AVG', padding='VALID', strides=[4, 4])
        # [4, 4]
        D_conv_6 = tf.layers.conv2d(D_mean_pool_5, filters=2, kernel_size=[
                                    3, 3], padding='SAME', activation=activation_function)
        # [4, 4]
        D_mean_pool_6 = tf.nn.pool(D_conv_6, window_shape=[
                                   4, 4], pooling_type='AVG', padding='VALID', strides=[4, 4])
        # [1, 1], and finally, [batch_size][1][1][2]
        D_logit = tf.reshape(D_mean_pool_6, shape=[128, 2])
        # [batch_size][2]

        return D_logit

        '''
        D_hidden_layer_1 = tf.layers.dense(
            inputs=x, units=255, activation=activation_function)
        D_hidden_layer_2 = tf.layers.dense(
            inputs=D_hidden_layer_1, units=16, activation=activation_function)
        D_logit = tf.layers.dense(inputs=D_hidden_layer_2, units=Y_dim,
                                  activation=activation_function)

        return D_logit
        '''


def generator(z, name=None):
    with tf.name_scope(name, "generator", [z]) as scope:
        # z[128, 4096]
        input = tf.reshape(z, shape=[128, 256, 256, 1])
        # input[128, 64, 64, 1]
        G_conv_1 = tf.layers.conv2d(input, filters=96, kernel_size=[
                                    8, 8], padding='SAME', activation=activation_function)
        # [128, 64, 64, 96]
        # G_upscaled_1 = tf.image.resize_bicubic(images=G_conv_1, size=[128, 128])
        # [128, 128, 128, 96]
        G_conv_2 = tf.layers.conv2d(G_conv_1, filters=64, kernel_size=[
                                    5, 5], padding='SAME', activation=activation_function)
        # [128, 128, 128, 64]
        # G_upscaled_2 = tf.image.resize_bicubic(G_conv_2, size=[256, 256])
        # [128, 256, 256, 64]
        G_conv_3 = tf.layers.conv2d(G_conv_2, filters=64, kernel_size=[
                                    5, 5], padding='SAME', activation=activation_function)
        # [128, 256, 256, 64]
        G_conv_4 = tf.layers.conv2d(G_conv_3, filters=1, kernel_size=[
                                    5, 5], padding='SAME', activation=activation_function)
        # [128, 256, 256, 1]
        G_logit = G_conv_4 * custom_filter
        # [128, 256, 256, 1], but filtered out the last column and row

        return G_logit

        '''
        G_hidden_layer_1 = tf.layers.dense(
            inputs=z, units=255, activation=activation_function)
        G_outputs = tf.layers.dense(inputs=G_hidden_layer_1, units=X_dim,
                                    activation=activation_function)

        return G_outputs
        '''


with tf.name_scope("training") as scope:
    # Getting samples from random data
    G_sample = generator(Z)
    # Getting logits
    D_logit_real = discriminator(X)
    D_logit_fake = discriminator(G_sample)
    # Applying softmax
    D_proba_real = tf.nn.softmax(logits=D_logit_real)
    D_proba_fake = tf.nn.softmax(logits=D_logit_fake)

    with tf.name_scope("category_1") as sub_scope:
        # Getting Shannon's entrophy in X's distribution
        D_log_real = tf.log(D_proba_real)
        D_entrophy_real = -(D_proba_real * D_log_real)
        D_mean_real = tf.reduce_mean(D_entrophy_real, axis=1)
        D_entrophy_real_mean = tf.reduce_mean(D_mean_real, axis=0)
        D_entrophy_real_mean = tf.reshape(D_entrophy_real_mean, shape=[1])

    with tf.name_scope("category_2") as sub_scope:
        # Gettning Shannon's entrophy in Z's distribution
        G_log_fake = tf.log(D_proba_fake)
        G_entrophy_fake = -(D_proba_fake * G_log_fake)
        G_mean = tf.reduce_mean(G_entrophy_fake, axis=1)
        G_entrophy_fake_mean = tf.reduce_mean(G_mean, axis=0)
        G_entrophy_fake_mean = tf.reshape(G_entrophy_fake_mean, shape=[1])

    with tf.name_scope("category_3") as sub_scope:
        # Getting Shannon's entrophy between classes
        D_class_mean = tf.reduce_mean(D_entrophy_real, axis=0, keep_dims=True)
        D_class = tf.reduce_mean(D_class_mean, axis=1)

        G_class_mean = tf.reduce_mean(G_entrophy_fake, axis=0, keep_dims=True)
        G_class = tf.reduce_mean(G_class_mean, axis=1)
        G_class = tf.reshape(G_class, shape=[1])

    with tf.name_scope("supervised") as sub_scope:
        # Getting cross entrophy for labeled data
        D_labeled = Y * D_log_real
        D_cross_entrophy = tf.reduce_mean(D_labeled, axis=1)
        D_cross_entrophy = -D_cross_entrophy
        D_supervised = tf.reduce_mean(D_cross_entrophy, axis=0)
        D_supervised_weighted = value_lambda * D_supervised
        D_supervised_weighted = tf.reshape(D_supervised_weighted, shape=[1])

    D_loss = D_class - D_entrophy_real_mean + \
        G_entrophy_fake_mean + D_supervised_weighted
    G_loss = G_class + G_entrophy_fake_mean
    D_loss = -D_loss

    D_solver = tf.train.AdamOptimizer().minimize(D_loss)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss)

# with tf.name_scope("testing") as scope:
