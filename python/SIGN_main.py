import tensorflow as tf
import numpy as np
from SIGN_file_reader import SIGN_file_reader
from SIGN_model import model
from PIL import Image

tf.logging.set_verbosity(verbosity=tf.logging.ERROR)


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
        tp_x_batch = sess.run(train_x_pos_batch)
        tn_x_batch = sess.run(train_x_neg_batch)
        tep_x_batch = sess.run(test_x_pos_batch)
        ten_x_batch = sess.run(test_x_neg_batch)

        coord.request_stop()
        coord.join(threads)

    '''
    Image.fromarray(np.reshape(first, newshape=[255, 255]), mode="F").show()
    Image.fromarray(np.reshape(second, newshape=[255, 255]), mode="F").show()
    '''


if __name__ == "__main__":
    main()
