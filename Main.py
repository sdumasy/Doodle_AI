from preprocess.ImgLoader import ImgLoader
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorlayer as tl
import time
import DataContainer as dc
import logging
from tensorlayer.layers import *
from Constants import Constants
from sklearn.preprocessing import normalize

C = Constants()

loader = ImgLoader(C.train_record_output_dir)
# loader.count_images_per_class()
# loader.create_image_records()
# loader.rec_img(output_file + '/record_4')

data_container = dc.DataContainer(C.record_output_dir)
# data_container.visualise_data(True)
tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


def model(x, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        network = tl.layers.InputLayer(x, name='input')

        ## Simplified conv API (the same with the above layers)
        network = tl.layers.Conv2d(network, 32, (20, 20), (2, 2), act=tf.nn.relu, padding='VALID', name='cnn1')
        network = tl.layers.MaxPool2d(network, (3, 3), (2, 2), padding='VALID', name='pool1')
        network = tl.layers.Conv2d(network, 64, (5, 5), (2, 2), act=tf.nn.relu, padding='VALID', name='cnn2')
        network = tl.layers.MaxPool2d(network, (2, 2), (2, 2), padding='VALID', name='pool2')
        network = tl.layers.Conv2d(network, 64, (2, 2), (2, 2), act=tf.nn.relu, padding='VALID', name='cnn3')
        network = tl.layers.MaxPool2d(network, (2, 2), (2, 2), padding='VALID', name='pool3')
        ## end of conv
        network = tl.layers.FlattenLayer(network, name='flatten')
        # network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1', is_fix = is_train)
        network = tl.layers.DenseLayer(network, 256, act=tf.nn.relu, name='relu1')
        # network = tl.layers.DropoutLayer(network, keep=0.9, name='drop2', is_fix = is_train)
        network = tl.layers.DenseLayer(network, 100, act=tf.nn.relu, name='relu2')
        # network = tl.layers.DropoutLayer(network, keep=0.8, name='drop3', is_fix = is_train)
        network = tl.layers.DenseLayer(network, C.output_classes, act=tf.identity, name='output')
        return network


def train():

    sess = tf.InteractiveSession()
    n_epoch = 1000
    n_step_epoch = int(C.num_training / C.batch_size)

    y_batch = data_container.y_train_batch
    x_batch = data_container.x_train_batch
    k_batch = data_container.key_train_batch

    # y_dev_batch = data_container.y_dev_batch
    # x_dev_batch = data_container.x_dev_batch

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_network = model(x_batch)
    # test_network = model(x_dev_batch, is_train=False, reuse=True)


    # Training cost
    y = train_network.outputs

    # train_cost = -tf.reduce_sum(tf.one_hot(y_batch, C.output_classes)*tf.log(tf.nn.softmax(y) + 1e-10))


    # train_cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_batch)
    train_cost = tl.cost.cross_entropy(y, y_batch, name='xentropy')
    # train_cost = -tf.reduce_sum(y_batch*tf.log(tf.clip_by_value(y,1e-10,1.0)))

    correct_train_prediction = tf.equal(tf.argmax(y, 1), y_batch)
    acc_train = tf.reduce_mean(tf.cast(correct_train_prediction, tf.float32))

    # Evaluation cost
    # y2 = test_network.outputs
    # cost_test = tl.cost.cross_entropy(y2, y_dev_batch, name='xentropy2')
    # correct_test_prediction = tf.equal(tf.argmax(y2, 1), y_dev_batch)
    # acc_test = tf.reduce_mean(tf.cast(correct_test_prediction, tf.float32))

    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(train_cost)
    # train_op = tf.train.AdamOptimizer(5e-5, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(train_cost)
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epoch):
        print("EPOCH NUM:", epoch)
        start_time = time.time()
        train_loss, train_acc, n_batch = 0, 0, 0
        for s in range(n_step_epoch):
            err, ac, _ = sess.run([train_cost, acc_train, train_op])
            # print("ERROR", err, "ACC", ac)
            train_loss += err
            train_acc += ac
            n_batch += 1

            # if s % 50 == 0:
            #     print("Epoch:", epoch, "of", n_epoch, "Step number:", str(s), "of", n_step_epoch)

        print("Epoch took:", time.time() - start_time)
        print("Train loss:" , (train_loss / n_batch))
        print("Training accuracy", (train_acc / n_batch))

        # dev_loss, dev_acc, n_batch = 0, 0, 0
        # for s in range(n_step_epoch):
        #     err, ac = sess.run([cost_test, acc_test])
        #     dev_loss += err
        #     dev_acc += ac
        #     n_batch += 1
        #
        # print("Dev loss:" , (dev_loss / n_batch))
        # print("Dev accuracy", (dev_acc / n_batch))

    coord.request_stop()
    coord.join(threads)
    sess.close()

train()
