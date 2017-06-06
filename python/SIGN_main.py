import tensorflow as tf
import numpy as np
import SIGN_model
from SIGN_file_reader import SIGN_file_reader
from SIGN_aux import sample_Z
from PIL import Image

tf.logging.set_verbosity(verbosity=tf.logging.ERROR)

batch_size = 128
iteration = 1000


def main():
    # getting a reader
    reader = SIGN_file_reader()
    # getting producers
    with tf.variable_scope("SIGN", reuse=True) as scope:
        train_pos_producer = reader.get_producer("producer:train:pos")
        train_neg_producer = reader.get_producer("producer:train:neg")
        test_pos_producer = reader.get_producer("producer:test:pos")
        test_neg_producer = reader.get_producer("producer:test:neg")

        train_pos_size = reader.get_size("size:train:pos")
        train_neg_size = reader.get_size("size:train:neg")
        test_pos_size = reader.get_size("size:test:pos")
        test_neg_size = reader.get_size("size:test:neg")

        train_pos_reader = tf.WholeFileReader(name="reader/train/pos")
        train_neg_reader = tf.WholeFileReader(name="reader/train/neg")
        test_pos_reader = tf.WholeFileReader(name="reader/test/pos")
        test_neg_reader = tf.WholeFileReader(name="reader/test/neg")

        tp_key, tp_value = train_pos_reader.read(train_pos_producer)
        tn_key, tn_value = train_neg_reader.read(train_neg_producer)
        tep_key, tep_value = test_pos_reader.read(test_pos_producer)
        ten_key, ten_value = test_neg_reader.read(test_neg_producer)

        tp_image = tf.image.decode_jpeg(tp_value, channels=1)
        tp_image = tf.image.resize_images(tp_image, size=[257, 256])
        tn_image = tf.image.decode_jpeg(tn_value, channels=1)
        tn_image = tf.image.resize_images(tn_image, size=[257, 256])
        tep_image = tf.image.decode_jpeg(tep_value, channels=1)
        tep_image = tf.image.resize_images(tep_image, size=[257, 256])
        ten_image = tf.image.decode_jpeg(ten_value, channels=1)
        ten_image = tf.image.resize_images(ten_image, size=[257, 256])

        tp_label = tf.ones([1])
        tn_label = tf.zeros([1])
        tep_label = tf.ones([1])
        ten_label = tf.zeros([1])

        train_x_pos_batch = tf.train.batch([tp_image[0: -1]], batch_size=128)
        train_x_neg_batch = tf.train.batch([tn_image[0: -1]], batch_size=128)
        test_x_pos_batch = tf.train.batch([tep_image[0: -1]], batch_size=128)
        test_x_neg_batch = tf.train.batch([ten_image[0: -1]], batch_size=128)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for it in range(iteration):
        # train step
            # positive
            tp_x_batch = sess.run(train_x_pos_batch)
            x_batch_reshaped = tf.reshape(
                tensor=tp_x_batch, shape=[128, SIGN_model.X_dim])

            Z_sample = sample_Z(batch_size, SIGN_model.Z_dim)
            Y_label = np.zeros(shape=[batch_size, SIGN_model.Y_dim])
            Y_label[:, 1] = 1

            _, D_loss_current = sess.run([SIGN_model.D_solver, SIGN_model.D_loss], feed_dict={
                SIGN_model.X: x_batch_reshaped, SIGN_model.Y: Y_label, SIGN_model.Z: Z_sample})
            _, G_loss_current = sess.run([SIGN_model.G_solver, SIGN_model.G_loss], feed_dict={
                SIGN_model.Y: Y_label, SIGN_model.Z: Z_sample})

            if it % 100 == 0:
                print("Iteration : {}".format(it))
                print("Discriminator loss : {:.4}".format(D_loss_current))
                print("Generator loss : {:.4}".format(G_loss_current))
                print()

            # negative
            tn_x_batch = sess.run(train_x_neg_batch)

        # test step
            # positive
            tep_x_batch = sess.run(test_x_pos_batch)

            # negative
            ten_x_batch = sess.run(test_x_neg_batch)

            coord.request_stop()
            coord.join(threads)``

    '''
    Image.fromarray(np.reshape(first, newshape=[255, 255]), mode="F").show()
    Image.fromarray(np.reshape(second, newshape=[255, 255]), mode="F").show()
    '''


if __name__ == "__main__":
    main()
