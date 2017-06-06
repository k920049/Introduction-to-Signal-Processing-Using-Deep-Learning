import tensorflow as tf
import numpy as np

from SIGN_aux import sample_Z

# constants
X_dim = 256 * 256
Y_dim = 2
H_dim = 256
Z_dim = 256

# input placeholders
X = tf.placeholder(tf.float32, shape=[None, X_dim])
Y = tf.placeholder(tf.float32, shape=[None, Y_dim])
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

initializer = tf.contrib.layers.variance_scaling_initializer

with tf.variable_scope("SIGN") as scope:
    scope.reuse_variables()

    with tf.variable_scope("discriminator") as discriminator_scope:
        discriminator_scope.reuse_variables()

        discriminator_weight_1 = tf.get_variable(name="weight_1", shape=[
                                                 X_dim + Y_dim, H_dim], dtype=tf.float32, initializer=initializer)
        discriminator_b_1 = tf.get_variable(
            name="b_1", shape=[H_dim], dtype=tf.float32)

        discriminator_weight_2 = tf.get_variable(
            name="weight_2", shape=[H_dim, 1], dtype=tf.float32, initializer=initializer)
        discriminator_b_2 = tf.get_variable(
            name="b_2", shape=[1], dtype=tf.float32)

    with tf.variable_scope("generator") as generator_scope:
        generator_scope.reuse_variables()

        generator_weight_1 = tf.get_variable(name="weight_1", shape=[
            Z_dim + Y_dim, H_dim], dtype=tf.float32, initializer=initializer)
        generator_b_1 = tf.get_variable(
            name="b_1", shape=[H_dim], dtype=tf.float32)

        generator_weight_2 = tf.get_variable(
            name="weight_2", shape=[H_dim, X_dim], dtype=tf.float32, initializer=initializer)
        generator_b_2 = tf.get_variable(
            name="b_2", shape=[X_dim], dtype=tf.float32)

theta_D = [discriminator_weight_1, discriminator_weight_2,
           discriminator_b_1, discriminator_b_2]

theta_G = [generator_weight_1, generator_weight_2,
           generator_b_1, generator_b_2]


def discriminator(x, y):

    inputs = tf.concat(axis=1, values=[x, y])
    discriminator_hypotheses = tf.matmul(
        inputs, discriminator_weight_1) + discriminator_b_1
    discriminator_activation = tf.nn.elu(features=discriminator_hypotheses)
    discriminator_logit = tf.matmul(
        discriminator_activation, discriminator_weight_2) + discriminator_b_2
    discriminator_proba = tf.nn.sigmoid(discriminator_logit)

    return discriminator_proba, discriminator_logit


def generator(z, y):

    inputs = tf.concat(axis=1, values=[z, y])
    generator_hypotheses = tf.matmul(
        inputs, generator_weight_1) + generator_b_1
    generator_activation = tf.nn.elu(features=generator_hypotheses)
    generator_logit = tf.matmul(
        generator_activation, generator_weight_2) + generator_b_2
    generator_proba = tf.nn.sigmoid(generator_logit)

    return generator_proba


with tf.name_scope("training") as scope:
    G_sample = generator(Z, Y)

    D_real, D_logit_real = discriminator(X, Y)
    D_fake, D_logit_fake = discriminator(G_sample, Y)

    # tell the network it's the genuine
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    # tell the network it's fake
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    # add the two losses
    D_loss = D_loss_real + D_loss_fake
    # only take into account the loss of the fake one
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

    # minimize the descriminator's loss
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    # minimize the generator's loss
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

with tf.name_scope("classification") as scope:
