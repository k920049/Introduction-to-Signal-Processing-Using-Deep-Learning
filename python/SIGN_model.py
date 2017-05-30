import tensorflow as tf

initializer = tf.contrib.layers.variance_scaling_initializer
activation_fn = tf.nn.elu

def model(scope_name, input):
    with tf.variable_scope(scope_name) as scope:
        scope.reuse_variables()

        conv1 = tf.layers.conv2d(inputs=input, kernel_size=(
            4, 4), filters=1024, padding='same', activation=activation_fn, kernel_initializer=initializer, name="conv1")

        conv2 = tf.layers.conv2d(inputs=conv1, kernel_size=(
            8, 8), filters=512, padding='same', activation=activation_fn, kernel_initializer=initializer, name='conv2')

        conv3 = tf.layers.conv2d(inputs=conv2, kernel_size=(16, 16), filters=256, padding='same',
                                 activation=activation_fn, kernel_initializer=initializer, name='conv3')

        conv4 = tf.layers.conv2d(inputs=conv3, kernel_size=(32, 32), filters=128, padding='same',
                                 activation=activation_fn, kernel_initializer=initializer, name='conv4')

        conv5 = tf.layers.conv2d(inputs=conv4, kernel_size=(
            64, 64), filters=3, padding='same', activation=activation_fn, kernel_initializer=initializer, name='conv5')

        return conv5
