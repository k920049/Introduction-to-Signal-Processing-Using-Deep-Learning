import tensorflow as tf
import numpy as np

# constants
X_dim = 255 * 255
Y_dim = 2
Z_dim = 64
value_lambda = 1.0

X = tf.placeholder(tf.float32, shape=[None, X_dim])
Y = tf.placeholder(tf.float32, shape=[None, Y_dim])
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

initializer = tf.contrib.layers.variance_scaling_initializer
activation_function = tf.nn.elu


def discriminator(x, name=None):
    with tf.name_scope(name, "discriminator", [x]) as scope:

        D_hidden_layer_1 = tf.layers.dense(
            inputs=x, units=255, activation=activation_function)
        D_hidden_layer_2 = tf.layers.dense(
            inputs=D_hidden_layer_1, units=16, activation=activation_function)
        D_logit = tf.layers.dense(inputs=D_hidden_layer_2, units=Y_dim,
                                  activation=activation_function)

        return D_logit


def generator(z, name=None):
    with tf.name_scope(name, "generator", [z]) as scope:

        G_hidden_layer_1 = tf.layers.dense(
            inputs=z, units=255, activation=activation_function)
        G_outputs = tf.layers.dense(inputs=G_hidden_layer_1, units=X_dim,
                                    activation=activation_function)

        return G_outputs


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

    with tf.name_scope("category_2") as sub_scope:
        # Gettning Shannon's entrophy in Z's distribution
        G_log_fake = tf.log(D_proba_fake)
        G_entrophy_fake = -(D_proba_fake * G_log_fake)
        G_mean = tf.reduce_mean(G_entrophy_fake, axis=1)
        G_entrophy_fake_mean = tf.reduce_mean(G_mean, axis=0)

    with tf.name_scope("category_3") as sub_scope:
        # Getting Shannon's entrophy between classes
        D_class_mean = tf.reduce_mean(D_entrophy_real, axis=0, keep_dims=True)
        D_class = tf.reduce_mean(D_class_mean, axis=1)

        G_class_mean = tf.reduce_mean(G_entrophy_fake, axis=0, keep_dims=True)
        G_class = tf.reduce_mean(G_class_mean, axis=1)

    with tf.name_scope("supervised") as sub_scope:
        # Getting cross entrophy for labeled data
        D_labeled = Y * D_log_real
        D_cross_entrophy = tf.reduce_mean(D_labeled, axis=1)
        D_cross_entrophy = -D_cross_entrophy
        D_supervised = tf.reduce_mean(D_cross_entrophy, axis=0)
        D_supervised_weighted = value_lambda * D_supervised

    D_loss = D_class - D_entrophy_real_mean + \
        G_entrophy_fake_mean + D_supervised_weighted
    G_loss = G_class + G_entrophy_fake_mean
    D_loss = -D_loss

    D_solver = tf.train.AdamOptimizer().minimize(D_loss)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss)

# with tf.name_scope("testing") as scope:
