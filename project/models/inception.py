import tensorflow as tf
import numpy as np


class InceptionV3:
    def __init__(self, X, num_outputs):
        self.init = tf.contrib.layers.variance_scaling_initializer
        self.act = tf.nn.elu
        self.logit = None
        self.base = X
        self.num_outputs = num_outputs

        return self._build_network()

    def _build_network(self):
        with tf.name_scope("network"):
            with tf.name_scope("conv1"):
                self.base = tf.layers.conv2d(inputs=self.base, filters=4, kernel_size=(1, 3, 3, 1), strides=(
                    1, 1, 1, 1), padding='same', activation=self.act, kernel_initializer=self.init)
                self.base = tf.layers.max_pooling2d(inputs=self.base, pool_size=(
                    1, 3, 3, 1), strides=(1, 2, 2, 1), padding='valid')
            with tf.name_scope("conv2"):
                self.base = tf.layers.conv2d(inputs=self.base, filters=8, kernel_size=(1, 3, 3, 1), strides=(
                    1, 2, 2, 1), padding='valid', activation=self.act, kernel_initializer=self.init)
            self.base = self.inception_layer_1(self.base, 16)
            self.base = self.inception_layer_2(self.base, 32)
            self.base = self.inception_layer_3(self.base, 64)
            with tf.name_scope("pool"):
                self.base = tf.layers.max_pooling2d(inputs=self.base, pool_size=(
                    1, 7, 7, 1), strides=(1, 1, 1, 1), padding='valid')
            with tf.name_scope("dense"):
                self.base = tf.layers.flatten(self.base)
                self.base = tf.layers.dense(
                    inputs=self.base, units=self.num_outputs, kernel_initializer=self.init)
        self.logit = self.base
        return self.logit

    def inception_layer_1(self, base, num_layers):
        with tf.name_scope("inception_layer_1"):
            base1 = base
            base2 = base
            base3 = base
            base4 = base
            with tf.name_scope("conv1"):
                base1 = tf.layers.conv2d(inputs=base1, filter=(num_layers / 4), kernel_size=(1, 1, 1, 1), strides=(
                    1, 1, 1, 1), padding='same', activation=self.act, kernel_initializer=self.init)
                base1 = tf.layers.conv2d(inputs=base1, filter=(num_layers / 4), kernel_size=(1, 3, 3, 1), strides=(
                    1, 1, 1, 1), padding='same', activation=self.act, kernel_initializer=self.init)
                base1 = tf.layers.conv2d(inputs=base1, filter=(num_layers / 4), kernel_size=(1, 3, 3, 1), strides=(
                    1, 2, 2, 1), padding='valid', activation=self.act, kernel_initializer=self.init)
            with tf.name_scope("conv2"):
                base2 = tf.layers.conv2d(inputs=base2, filter=(num_layers / 4), kernel_size=(1, 1, 1, 1), strides=(
                    1, 1, 1, 1), padding='same', activation=self.act, kernel_initializer=self.init)
                base2 = tf.layers.conv2d(inputs=base2, filter=(num_layers / 4), kernel_size=(1, 3, 3, 1), strides=(
                    1, 2, 2, 1), padding='valid', activation=self.act, kernel_initializer=self.init)
            with tf.name_scope("pool"):
                base3 = tf.layers.max_pooling2d(
                    inputs=base3, pool_size=(1, 2, 2, 1), strides=(1, 2, 2, 1))
                base3 = tf.layers.conv2d(inputs=base3, filter=(num_layers / 4), kernel_size=(1, 1, 1, 1), strides=(
                    1, 1, 1, 1), padding='same', activation=self.act, kernel_initializer=self.init)
            with tf.name_scope("conv4"):
                base4 = tf.layers.conv2d(inputs=base4, filter=(num_layers / 4), kernel_size=(1, 3, 3, 1), strides=(
                    1, 2, 2, 1), padding='valid', activation=self.act, kernel_initializer=self.init)
        return tf.concat([base1, base2, base3, base4])

    def inception_layer_2(self, base, num_layers):
        with tf.name_scope("inception_layer_2"):
            base1 = base
            base2 = base
            base3 = base
            base4 = base
            with tf.name_scope("conv1"):
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(1, 1, 1, 1), strides=(
                    1, 1, 1, 1), padding='same', activation=self.act, kernel_initializer=self.init)
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(1, 1, 7, 1), strides=(
                    1, 1, 1, 1), padding='same', activation=self.act, kernel_initializer=self.init)
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(1, 7, 1, 1), strides=(
                    1, 1, 1, 1), padding='same', activation=self.act, kernel_initializer=self.init)
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(1, 1, 7, 1), strides=(
                    1, 1, 1, 1), padding='same', activation=self.act, kernel_initializer=self.init)
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(1, 7, 1, 1), strides=(
                    1, 2, 2, 1), padding='valid', activation=self.act, kernel_initializer=self.init)
            with tf.name_scope("conv2"):
                base2 = tf.layers.conv2d(inputs=base2, filters=(num_layers / 4), kernel_size=(1, 1, 1, 1), strides=(
                    1, 1, 1, 1), padding='same', activation=self.act, kernel_initializer=self.init)
                base2 = tf.layers.conv2d(inputs=base2, filters=(num_layers / 4), kernel_size=(1, 1, 7, 1), strides=(
                    1, 1, 1, 1), padding='same', activation=self.act, kernel_initializer=self.init)
                base2 = tf.layers.conv2d(inputs=base2, filters=(num_layers / 4), kernel_size=(1, 7, 1, 1), strides=(
                    1, 2, 2, 1), padding='valid', activation=self.act, kernel_initializer=self.init)
            with tf.name_scope("pool"):
                base3 = tf.layers.max_pooling2d(inputs=base3, pool_size=(
                    1, 2, 2, 1), strides=(1, 2, 2, 1), padding='valid')
                base3 = tf.layers.conv2d(inputs=base3, filters=(num_layers / 4), kernel_size=(1, 1, 1, 1), strides=(
                    1, 1, 1, 1), padding='same', activation=self.act, kernel_initializer=self.init)
            with tf.name_scope("conv4"):
                base4 = tf.layers.conv2d(inputs=base4, filters=(num_layers / 4), kernel_size=(1, 1, 1, 1), strides=(
                    1, 2, 2, 1), padding='valid', activation=self.act, kernel_initializer=self.init)
        return tf.concat([base1, base2, base3, base4])

    def inception_layer_3(self, base, num_layers):
        with tf.name_scope("inception_layer_3"):
            base1 = base
            base2 = base
            base3 = base
            base4 = base
            with tf.name_scope("conv1"):
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(1, 1, 1, 1), strides=(
                    1, 1, 1, 1), padding='same', activation=self.act, kernel_initializer=self.init)
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(1, 3, 3, 1), strides=(
                    1, 1, 1, 1), padding='same', activation=self.act, kernel_initializer=self.init)
                sub_base1_1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 8), kernel_size=(
                    1, 1, 3, 1), strides=(1, 2, 2, 1), padding='valid', activation=self.act, kernel_initializer=self.init)
                sub_base1_2 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 8), kernel_size=(
                    1, 3, 1, 1), strides=(1, 2, 2, 1), padding='valid', activation=self.act, kernel_initializer=self.init)
                base1 = tf.concat([sub_base1_1, sub_base1_2])
            with tf.name_scope("conv2"):
                base2 = tf.layers.conv2d(inputs=base2, filters=(num_layers / 4), kernel_size=(1, 1, 1, 1), strides=(
                    1, 1, 1, 1), padding='same', activation=self.act, kernel_initializer=self.init)
                sub_base2_1 = tf.layers.conv2d(inputs=base2, filters=(num_layers / 8), kernel_size=(
                    1, 1, 3, 1), strides=(1, 2, 2, 1), padding='valid', activation=self.act, kernel_initializer=self.init)
                sub_base2_2 = tf.layers.conv2d(inputs=base2, filters=(num_layers / 8), kernel_size=(
                    1, 3, 1, 1), strides=(1, 2, 2, 1), padding='valid', activation=self.act, kernel_initializer=self.init)
                base2 = tf.concat([sub_base2_1, sub_base2_2])
            with tf.name_scope("pool"):
                base3 = tf.layers.max_pooling2d(inputs=base3, pool_size=(
                    1, 2, 2, 1), strides=(1, 2, 2, 1), padding='valid')
            with tf.name_scope("conv4"):
                base4 = tf.layers.conv2d(inputs=base4, filters=(num_layers / 4), kernel_size=(1, 1, 1, 1), strides=(
                    1, 2, 2, 1), padding='valid', activation=self.act, kernel_initializer=self.init)
        return tf.concat([base1, base2, base3, base4])

    def get_logit(self):
        return self.logit
