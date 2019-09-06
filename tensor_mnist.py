from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

INPUT_LAYER_SIZE = 28 * 28
HIDDEN_LAYER_SIZE = 500
OUTPUT_LAYER_SIZE = 10
# cnn
FULL_SIZE = 512
IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV1_DEEP = 32
CONV2_SIZE = 5
CONV2_DEEP = 64
# basic
batch_size = 100
LR_base = 0.1
LR_decay = 0.99
L2_rate = 0.0001
moving_average_decay = 0.99
training_step = 50000


def fcnet(input_tensor, regularizer):
    with tf.variable_scope("layer1"):
        weight = tf.get_variable('weight', [INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE], tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', [1, HIDDEN_LAYER_SIZE], tf.float32, initializer=tf.constant_initializer(0.1))
        hidden_layer = tf.nn.relu(tf.add(tf.matmul(input_tensor, weight), bias))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weight))
    with tf.variable_scope('layer2'):
        weight = tf.get_variable('weight', [HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE], tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', [1, OUTPUT_LAYER_SIZE], tf.float32, initializer=tf.constant_initializer(0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weight))
        output_layer = tf.add(tf.matmul(hidden_layer, weight), bias)
        return output_layer


def leNet(input_tensor, keep_prob, regularizer):
    with tf.variable_scope("layer1-conv1"):
        weight = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, weight, [1, 1, 1, 1], 'SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias))
        pool1 = tf.nn.max_pool2d(relu1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    with tf.variable_scope("layer2-conv2"):
        weight = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, weight, [1, 1, 1, 1], 'SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias))
        pool2 = tf.nn.max_pool2d(relu2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    pool2_shape = pool2.get_shape().as_list()
    pool2_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    pool2_flat = tf.reshape(pool2, [-1, pool2_size])
    with tf.variable_scope("layer3-fc1"):
        weight = tf.get_variable("weight", [pool2_size, FULL_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias", [FULL_SIZE], initializer=tf.constant_initializer(0.1))
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(weight))
        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(pool2_flat, weight), bias))
        fc1 = tf.nn.dropout(fc1, keep_prob)
    with tf.variable_scope("layer4-fc2"):
        weight = tf.get_variable("weight", [FULL_SIZE, OUTPUT_LAYER_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias", [OUTPUT_LAYER_SIZE], initializer=tf.constant_initializer(0.1))
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(weight))
        fc2 = tf.nn.bias_add(tf.matmul(fc1, weight), bias)
    return fc2


def train(input_tensor):
    # init
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name='x-input')
    y = tf.placeholder(tf.float32, [None, OUTPUT_LAYER_SIZE], name='y-input')
    is_train = tf.placeholder(tf.bool, name='is_train')
    l2 = tf.contrib.layers.l2_regularizer(L2_rate)
    # forward
    # y_ = fcnet(x, l2)
    if is_train == 1:
        y_ = leNet(x, 0.5, l2)
    else:
        y_ = leNet(x, 1.0, None)
    global_step = tf.get_variable('global', initializer=0, trainable=False)
    variable_average = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    # loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_, labels=tf.argmax(y, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    if is_train == 1:
        loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    else:
        loss = cross_entropy_mean
    # optimizer
    lr = tf.train.exponential_decay(LR_base, global_step, input_tensor.train.num_examples / batch_size, LR_decay)
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
    train_op = tf.group(optimizer, variable_average_op)
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # start_train
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        val_x = np.reshape(input_tensor.validation.images, (-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        validation_feed = {
            x: val_x,
            y: input_tensor.validation.labels,
            is_train: 0
        }
        for i in range(training_step):
            if i % 1000 == 0:
                validation_acc = sess.run(acc, feed_dict=validation_feed)
                print("After %d training steps, validation accuracy :%g " %
                      (i, validation_acc))
            xs, ys = input_tensor.train.next_batch(batch_size)
            xs = np.reshape(xs, (batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
            sess.run(train_op, feed_dict={x: xs, y: ys, is_train: 1})
            if i % 100 == 0:
                loss_train = sess.run(loss, feed_dict={x: xs, y: ys, is_train: 1})
                print("After %d training steps, loss :%g " %
                      (i, loss_train))
        saver.save(sess, "./mnist_model")
    # start test
    with tf.Session() as sess:
        saver.restore(sess, "./mnist_model")
        test_x = np.reshape(input_tensor.test.images, (-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        test_feed = {
            x: test_x,
            y: input_tensor.test.labels,
            is_train: 0
        }
        test_acc = sess.run(acc, feed_dict=test_feed)
        print("Test accuracy : %g" % test_acc)
    saver = tf.train.Saver(variable_average.variables_to_restore())
    # test using average
    with tf.Session() as sess:
        saver.restore(sess, './mnist_model')
        test_x = np.reshape(input_tensor.test.images, (-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        test_feed = {
            x: test_x,
            y: input_tensor.test.labels,
            is_train: 0
        }
        test_acc = sess.run(acc, feed_dict=test_feed)
        print("Test average accuracy : %g" % test_acc)


def main(argv=None):
    mnist = input_data.read_data_sets("./", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
