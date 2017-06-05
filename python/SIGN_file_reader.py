import tensorflow as tf
import numpy as np
import sys
import os
import glob


class SIGN_file_reader(object):

    def __init__(self):
        self.__read_directory()

    def __read_directory(self):
        # environment variables
        cwd = os.getcwd()
        self.train_demo_pos_path = os.path.join(cwd, "../data/traindemo/pos")
        self.train_demo_neg_path = os.path.join(cwd, "../data/traindemo/neg")
        self.test_demo_pos_path = os.path.join(cwd, "../data/testdemo/pos")
        self.test_demo_neg_path = os.path.join(cwd, "../data/testdemo/neg")
        # file I/O
        self.train_pos_filelist = []
        self.train_neg_filelist = []
        self.test_pos_filelist = []
        self.test_neg_filelist = []

        self.train_pos_producer = None
        self.train_neg_producer = None
        self.test_pos_producer = None
        self.test_neg_producer = None

    def get_size(self, mode):
        # return the size of the list
        if (mode == "size:train:pos"):
            return len(self.train_pos_filelist)
        elif (mode == "size:train:neg"):
            return len(self.train_neg_filelist)
        elif (mode == "size:test:pos"):
            return len(self.test_pos_filelist)
        elif (mode == "size:test:neg"):
            return len(self.test_neg_filelist)
        else:
            print("Error : Invalid mode has been entered", file=sys.stderr)
            return None

    def get_producer(self, mode):
        # sends the correct producer
        if (mode == "producer:train:pos"):
            if (len(self.train_pos_filelist) == 0):
                for file in os.listdir(self.train_demo_pos_path):
                    if file.endswith(".jpg"):
                        self.train_pos_filelist.append(
                            os.path.join(self.train_demo_pos_path, file))

            if (self.train_pos_producer is None):
                list_to_tensor_tp = tf.convert_to_tensor(
                    self.train_pos_filelist)
                self.train_pos_producer = tf.train.string_input_producer(
                    list_to_tensor_tp, shuffle=True, name="train_pos")
                return self.train_pos_producer
            else:
                return self.train_pos_producer

        elif (mode == "producer:train:neg"):
            if (len(self.train_neg_filelist) == 0):
                for file in os.listdir(self.train_demo_neg_path):
                    if file.endswith(".jpg"):
                        self.train_neg_filelist.append(
                            os.path.join(self.train_demo_neg_path, file))

            if (self.train_neg_producer is None):
                list_to_tensor_tn = tf.convert_to_tensor(
                    self.train_neg_filelist)
                self.train_neg_producer = tf.train.string_input_producer(
                    list_to_tensor_tn, shuffle=True, name="train_neg")
                return self.train_neg_producer
            else:
                return self.train_neg_producer

        elif (mode == "producer:test:pos"):
            if (len(self.test_pos_filelist) == 0):
                for file in os.listdir(self.test_demo_pos_path):
                    if file.endswith(".jpg"):
                        self.test_pos_filelist.append(
                            os.path.join(self.test_demo_pos_path, file))

            if (self.test_pos_producer is None):
                list_to_tensor_tep = tf.convert_to_tensor(
                    self.test_pos_filelist)
                self.test_pos_producer = tf.train.string_input_producer(
                    list_to_tensor_tep, shuffle=True, name="test_pos")
                return self.test_pos_producer
            else:
                return self.test_pos_producer

        elif (mode == "producer:test:neg"):
            if (len(self.test_neg_filelist) == 0):
                for file in os.listdir(self.test_demo_neg_path):
                    if file.endswith(".jpg"):
                        self.test_neg_filelist.append(
                            os.path.join(self.test_demo_neg_path, file))

            if (self.test_neg_producer is None):
                list_to_tensor_ten = tf.convert_to_tensor(
                    self.test_neg_filelist)
                self.test_neg_producer = tf.train.string_input_producer(
                    list_to_tensor_ten, shuffle=True, name="test_neg")
                return self.test_neg_producer
            else:
                return self.test_neg_producer

        else:
            print("Error : Invalid mode has been entered", file=sys.stderr)
            return None
