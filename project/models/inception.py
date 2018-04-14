import tensorflow as tf
import numpy as np


class InceptionV3:
    def __init__(self, X, num_outputs):
        self.init = tf.contrib.layers.variance_scaling_initializer
        self.act = tf.nn.elu
        self.logit = None
        self.base = X
        self.num_outputs = num_outputs

        self._build_network()

    def _build_network(self):
        with tf.name_scope("network"):
            with tf.name_scope("conv1"):
                # [N, 255, 255, 4]
                self.base = tf.layers.conv2d(inputs=self.base, filters=4, kernel_size=(3, 3), strides=(
                    1, 1), padding='same', activation=self.act)
                # [N, 127, 127, 4]
                self.base = tf.layers.max_pooling2d(inputs=self.base, pool_size=(
                    3, 3), strides=(2, 2), padding='valid')
            with tf.name_scope("conv2"):
                # [N, 63, 63, 8]
                self.base = tf.layers.conv2d(inputs=self.base, filters=8, kernel_size=(3, 3), strides=(
                    2, 2), padding='valid', activation=self.act)
            # [N, 31, 31, 16]
            self.base = self.inception_layer_1(self.base, 16)
            # [N, 16, 16, 32]
            self.base = self.inception_layer_2(self.base, 32)
            # [N, 8, 8, 64]
            self.base = self.inception_layer_3(self.base, 64)
            with tf.name_scope("pool"):
                self.base = tf.layers.max_pooling2d(inputs=self.base, pool_size=(
                    8, 8), strides=(1, 1), padding='valid')
            with tf.name_scope("dense"):
                self.base = tf.layers.flatten(self.base)
                self.base = tf.layers.dense(
                    inputs=self.base, units=self.num_outputs)
        self.logit = self.base

    def get_logit(self):
        return self.logit

    def inception_layer_1(self, base, num_layers):
        with tf.name_scope("inception_layer_1"):
            base1 = base
            base2 = base
            base3 = base
            base4 = base
            with tf.name_scope("conv1"):
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(1, 1), strides=(
                    1, 1), padding='same', activation=self.act)
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(3, 3), strides=(
                    1, 1), padding='same', activation=self.act)
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(3, 3), strides=(
                    2, 2), padding='valid', activation=self.act)
            with tf.name_scope("conv2"):
                base2 = tf.layers.conv2d(inputs=base2, filters=(num_layers / 4), kernel_size=(1, 1), strides=(
                    1, 1), padding='same', activation=self.act)
                base2 = tf.layers.conv2d(inputs=base2, filters=(num_layers / 4), kernel_size=(3, 3), strides=(
                    2, 2), padding='valid', activation=self.act)
            with tf.name_scope("pool"):
                base3 = tf.layers.max_pooling2d(
                    inputs=base3, pool_size=(2, 2), strides=(2, 2))
                base3 = tf.layers.conv2d(inputs=base3, filters=(num_layers / 4), kernel_size=(1, 1), strides=(
                    1, 1), padding='same', activation=self.act)
            with tf.name_scope("conv4"):
                base4 = tf.layers.conv2d(inputs=base4, filters=(num_layers / 4), kernel_size=(3, 3), strides=(
                    2, 2), padding='valid', activation=self.act)
        return tf.concat([base1, base2, base3, base4], 3)

    def inception_layer_2(self, base, num_layers):
        with tf.name_scope("inception_layer_2"):
            base1 = base
            base2 = base
            base3 = base
            base4 = base
            with tf.name_scope("conv1"):
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(1, 1), strides=(
                    1, 1), padding='same', activation=self.act)
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(1, 7), strides=(
                    1, 1), padding='same', activation=self.act)
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(7, 1), strides=(
                    1, 1), padding='same', activation=self.act)
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(1, 7), strides=(
                    1, 1), padding='same', activation=self.act)
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(7, 1), strides=(
                    2, 2), padding='same', activation=self.act)
            with tf.name_scope("conv2"):
                base2 = tf.layers.conv2d(inputs=base2, filters=(num_layers / 4), kernel_size=(1, 1), strides=(
                    1, 1), padding='same', activation=self.act)
                base2 = tf.layers.conv2d(inputs=base2, filters=(num_layers / 4), kernel_size=(1, 7), strides=(
                    1, 1), padding='same', activation=self.act)
                base2 = tf.layers.conv2d(inputs=base2, filters=(num_layers / 4), kernel_size=(7, 1), strides=(
                    2, 2), padding='same', activation=self.act)
            with tf.name_scope("pool"):
                base3 = tf.layers.max_pooling2d(inputs=base3, pool_size=(
                    2, 2), strides=(2, 2), padding='same')
                base3 = tf.layers.conv2d(inputs=base3, filters=(num_layers / 4), kernel_size=(1, 1), strides=(
                    1, 1), padding='same', activation=self.act)
            with tf.name_scope("conv4"):
                base4 = tf.layers.conv2d(inputs=base4, filters=(num_layers / 4), kernel_size=(1, 1), strides=(
                    2, 2), padding='same', activation=self.act)
        return tf.concat([base1, base2, base3, base4], 3)

    def inception_layer_3(self, base, num_layers):
        with tf.name_scope("inception_layer_3"):
            base1 = base
            base2 = base
            base3 = base
            base4 = base
            with tf.name_scope("conv1"):
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(1, 1), strides=(
                    1, 1), padding='same', activation=self.act)
                base1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 4), kernel_size=(3, 3), strides=(
                    1, 1), padding='same', activation=self.act)
                sub_base1_1 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 8), kernel_size=(
                    1, 3), strides=(2, 2), padding='same', activation=self.act)
                sub_base1_2 = tf.layers.conv2d(inputs=base1, filters=(num_layers / 8), kernel_size=(
                    3, 1), strides=(2, 2), padding='same', activation=self.act)
                base1 = tf.concat([sub_base1_1, sub_base1_2], 3)
            with tf.name_scope("conv2"):
                base2 = tf.layers.conv2d(inputs=base2, filters=(num_layers / 4), kernel_size=(1, 1), strides=(
                    1, 1), padding='same', activation=self.act)
                sub_base2_1 = tf.layers.conv2d(inputs=base2, filters=(num_layers / 8), kernel_size=(
                    1, 3), strides=(2, 2), padding='same', activation=self.act)
                sub_base2_2 = tf.layers.conv2d(inputs=base2, filters=(num_layers / 8), kernel_size=(
                    3, 1), strides=(2, 2), padding='same', activation=self.act)
                base2 = tf.concat([sub_base2_1, sub_base2_2], 3)
            with tf.name_scope("pool"):
                base3 = tf.layers.max_pooling2d(inputs=base3, pool_size=(
                    2, 2), strides=(2, 2), padding='same')
                base3 = tf.layers.conv2d(inputs=base3, filters=(
                    num_layers / 4), kernel_size=(1, 1), strides=(1, 1), padding='same', activation=self.act)
            with tf.name_scope("conv4"):
                base4 = tf.layers.conv2d(inputs=base4, filters=(num_layers / 4), kernel_size=(1, 1), strides=(
                    2, 2), padding='same', activation=self.act)
        return tf.concat([base1, base2, base3, base4], 3)

    def get_logit(self):
        return self.logit
