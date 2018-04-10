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


def get_batch(image, class_index):
    batch_list = []
    for i in range(iteration[0]):
        image = sequence[0][3].eval()
        image = np.reshape(image, newshape=(width, height, 1))
        batch_list.append(image)

        if len(batch_list) == batch:
            X = np.stack(batch_list)
            y = np.ones(shape=(len(batch_list), 1), dtype=np.int32)
            y = class_index * y
            batch_list.clear()
            yield X, y
        elif i == len(iterations[0]) - 1:
            X = np.stack(batch_list)
            y = np.ones(shape=(len(batch_list), 1), dtype=np.int32)
            y = class_index * y
            batch_list.clear()
            yield X, y
        else:
            continue


for i in range(4):
    iterations.append(get_iteration(i))
    print(iterations[i])

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

X = tf.placeholder(dtype=tf.float32, shape=(batch, height, width, 1))
y = tf.placeholder(dtype=tf.int32, shape=(batch, 1))

inception_v3_logit = InceptionV3(X, 2)

with tf.name_scope("loss"):
    

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # Training sequence
    for batch_X, batch_y in get_batch(sequence[0][3]):


    coord.request_stop()
    coord.join(threads)
