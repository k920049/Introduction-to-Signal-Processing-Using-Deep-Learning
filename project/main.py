import tensorflow as tf
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import random
from PIL import Image

from os import listdir
from os.path import join, isfile

from models.inception import InceptionV3

# hyperparameters
epoch = 10
batch = 16
width = 255
height = 255
iterations = []

# set up data path
data_dir = "../data"

# training data
pos_train_path = join(data_dir, 'traindemo/pos/')
neg_train_path = join(data_dir, 'traindemo/neg/')

# validation data
pos_valid_path = join(data_dir, 'testdemo/pos/')
neg_valid_path = join(data_dir, 'testdemo/neg/')

# scope for register values
data_scope = {}
data_keys = ["pos_train",
             "neg_train",
             "pos_valid",
             "neg_valid"]
queue_keys = ["pos_train_queue",
              "neg_train_queue",
              "pos_valid_queue",
              "neg_valid_queue"]

pos_train = [join(pos_train_path, f) for f in listdir(
    pos_train_path) if isfile(join(pos_train_path, f))]
neg_train = [join(neg_train_path, f) for f in listdir(
    neg_train_path) if isfile(join(neg_train_path, f))]
pos_valid = [join(pos_valid_path, f) for f in listdir(
    pos_valid_path) if isfile(join(pos_valid_path, f))]
neg_valid = [join(neg_valid_path, f) for f in listdir(
    neg_valid_path) if isfile(join(neg_valid_path, f))]

data_scope["pos_train"] = pos_train
data_scope["neg_train"] = neg_train
data_scope["pos_valid"] = pos_valid
data_scope["neg_valid"] = neg_valid

# miscellaneous functions


def set_epoch(num_epoch):
    for i in range(len(data_keys)):
        data_scope[data_keys[i]] = data_scope[data_keys[i]] * num_epoch


def get_iteration(index):
    return len(data_scope[data_keys[i]])


def get_list(queue):
    reader = tf.WholeFileReader()
    key, value = reader.read(queue)
    img = tf.image.decode_jpeg(value)

    return reader, key, value, img

    set_epoch(epoch)

for i in range(4):
    iterations.append(get_iteration(i))

def get_batch(queue, index, class_index, total):
    batch_list = []
    for i in range(iterations[index]):
        image = queue.eval()
        image = np.reshape(image, newshape=(width, height, 1))
        batch_list.append(image)

        if len(batch_list) == batch:
            X = np.stack(batch_list)
            y = np.zeros(shape=(len(batch_list), total), dtype=np.float32)
            y[:, class_index] = 1
            y_label = np.ones(shape=(len(batch_list)), dtype=np.int32)
            y_label = y_label * class_index
            batch_list.clear()
            yield X, y, y_label
        elif i == iterations[index] - 1:
            X = np.stack(batch_list)
            y = np.zeros(shape=(len(batch_list), total), dtype=np.float32)
            y[:, class_index] = 1
            y_label = np.ones(shape=(len(batch_list)), dtype=np.int32)
            y_label = y_label * class_index
            batch_list.clear()
            yield X, y, y_label
        else:
            continue

pos_train_queue = tf.train.string_input_producer(data_scope["pos_train"])
neg_train_queue = tf.train.string_input_producer(data_scope["neg_train"])
pos_valid_queue = tf.train.string_input_producer(data_scope["pos_valid"])
neg_valid_queue = tf.train.string_input_producer(data_scope["neg_valid"])

data_scope["pos_train_queue"] = pos_train_queue
data_scope["neg_train_queue"] = neg_train_queue
data_scope["pos_valid_queue"] = pos_valid_queue
data_scope["neg_valid_queue"] = neg_valid_queue

sequence = []
for i in range(len(queue_keys)):
    reader, key, value, img = get_list(data_scope[queue_keys[i]])
    sequence.append([reader, key, value, img])

epsilon = tf.constant(0.1)
classes = 2

X = tf.placeholder(dtype=tf.float32, shape=(batch, height, width, 1))
Y = tf.placeholder(dtype=tf.float32, shape=(batch, classes))
y_label = tf.placeholder(dtype=tf.int32, shape=(batch))

I = InceptionV3(X, 2)
inception_v3_logit = I.get_logit()

save_interval = 10

with tf.name_scope("accuracy"):
    prob = tf.contrib.layers.softmax(inception_v3_logit)
    correct = tf.nn.in_top_k(inception_v3_logit, y_label, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("loss"):
    q = (1 - epsilon) * Y + (epsilon / classes)
    l_prob = tf.log(prob)
    div = -tf.multiply(l_prob, q)
    sum = tf.reduce_sum(div, 1)
    loss = tf.reduce_mean(sum, 0)

    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    count = 0
    # Training sequence
    for batch_X, batch_y, batch_label in get_batch(sequence[0][3], 0, 1, classes):
        sess.run(training_op, feed_dict={X: batch_X, Y: batch_y, y_label: batch_label})
        acc_train = accuracy.eval(feed_dict={X: batch_X, Y: batch_y, y_label: batch_label})
        print("Batch : " + str(count) + ", Accuracy : " + str(acc_train))
        count = count + 1
        if count % save_interval == 0:
            save_path = saver.save(sess, "./save/my_mnist_model")
    count = 0
    # Training sequence
    for batch_X, batch_y in get_batch(sequence[1][3], 1, 0, classes):
        sess.run(training_op, feed_dict={X: batch_X, Y: batch_y, y_label: batch_label})
        acc_train = accuracy.eval(feed_dict={X: batch_X, Y: batch_y, y_label: batch_label})
        print("Batch : " + str(count) + ", Accuracy : " + str(acc_train))
        count = count + 1
        if count % save_interval == 0:
            save_path = saver.save(sess, "./save/my_mnist_model")
    count = 0
    # Test sequence
    for batch_X, batch_y in get_batch(sequence[2][3], 2, 1, classes):
        acc_train = accuracy.eval(feed_dict={X: batch_X, Y: batch_y, y_label: batch_label})
        print("Batch : " + str(count) + ", Accuracy : " + str(acc_train))
        count = count + 1
    count = 0
    # Test sequence
    for batch_X, batch_y in get_batch(sequence[3][3], 3, 0, classes):
        acc_train = accuracy.eval(feed_dict={X: batch_X, Y: batch_y, y_label: batch_label})
        print("Batch : " + str(count) + ", Accuracy : " + str(acc_train))
        count = count + 1

    coord.request_stop()
    coord.join(threads)
