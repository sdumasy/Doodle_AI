import tensorflow as tf
import tensorlayer as tl
import time
import numpy as np
from preprocess import ImgLoader
from Constants import Constants
import glob
from PIL import Image


from matplotlib import pyplot as plt

class DataContainer(object):

    def __init__(self, input_dir):
        """Initiliase the data container with the feature names and amount of samples"""
        self.record_image = 'image_raw'
        self.record_label = 'class'
        self.record_key = 'key'
        self.C = Constants()
        self.min_after_dequeue = 1280
        self.threads = 32
        self.init_batches(input_dir)

    def append_in_folder(self, input_dir):
        """Appends all the files from a specific folder into a list"""
        return glob.glob(input_dir)

    def init_batches(self, input_dir):
        """Init the variables holding the queue with batchees"""
        print(input_dir+'train')
        self.train_files = self.append_in_folder(input_dir + 'train/record_5')
        self.dev_files = self.append_in_folder(input_dir + 'dev/*')
        self.test_files = self.append_in_folder(input_dir + 'test/*')
        print("De files", self.train_files)

        #Store all the data in the queue calling the variable will dequeueu once
        self.key_train_batch, self.x_train_batch, self.y_train_batch = self.parse_tf_rec(self.train_files, True)
        # self.key_dev_batch, self.x_dev_batch, self.y_dev_batch = self.parse_tf_rec(self.dev_files, True)
        # self.x_test_batch, self.y_test_batch = self.parse_tf_rec(self.test_files, True)

    def parse_tf_rec(self, file_names, is_train = None):
        """Parse the tfreocrds and store them into a queue with batches"""

        capacity = self.min_after_dequeue + (self.threads + 1) * self.C.batch_size
        filename_queue = tf.train.string_input_producer(file_names, shuffle=True) #Shuffled queue with filenames

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        if is_train:
            features = tf.parse_single_example(
                serialized_example, features={
                    self.record_key: tf.FixedLenFeature([], tf.string),
                    self.record_label : tf.FixedLenFeature([], tf.int64),
                    self.record_image : tf.FixedLenFeature([], tf.string),
                })
        else:
            features = tf.parse_single_example(
                serialized_example, features={
                    self.record_key : tf.FixedLenFeature([], tf.string),
                    self.record_image : tf.FixedLenFeature([], tf.string),
                })

        img = tf.decode_raw(features[self.record_image], tf.uint8)
        img = tf.reshape(img, [self.C.image_height, self.C.image_width, 1])
        # img = tf.to_float(img)
        img = tf.image.per_image_standardization(img)
        # img = tf.nn.l2_normalize(img)
        # img = tf.contrib.layers.instance_norm(img, epsilon=1e-06)
        # img = tf.image.per_image_standardization(img)

        key = tf.cast(features[self.record_key], tf.string)

        if(is_train):
            label = tf.cast(features[self.record_label], tf.int64)
            return tf.train.shuffle_batch([key, img, label], batch_size=self.C.batch_size, capacity=capacity, num_threads=self.threads, min_after_dequeue=self.min_after_dequeue)
        else:
            return tf.train.shuffle_batch([key, img], batch_size=self.C.batch_size, capacity=capacity, num_threads=self.threads, min_after_dequeue=self.min_after_dequeue)

    def visualise_data(self, train_set = False):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(10):
                print("Step %d" % i)
                if(train_set):
                    key, val, l = sess.run([self.key_train_batch, self.x_train_batch, self.y_train_batch])
                    print(self.C.class_list[l[0]])
                    print(val[0][0])
                    plt.imshow(val[0].reshape(255,255), interpolation='nearest')
                    plt.show()
                # else:
                #     key, val = sess.run([self.key_test_batch, self.x_test_batch])
                # self.show_image(val[i])

            coord.request_stop()
            coord.join(threads)
            sess.close()