import tensorflow as tf
from tensorflow.contrib.layers import flatten

def simple_convnet(x, mu, sigma):

    conv_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv_b = tf.Variable(tf.zeros(6))

    conv = tf.nn.conv2d(x, conv_W, strides=[1, 1, 1, 1], padding="VALID") + conv_b

    conv = tf.nn.relu(conv)

    fc = flatten(conv)

    fc_W = tf.Variable(tf.truncated_normal(shape=(4704, 10), mean=mu, stddev=sigma))
    fc_b = tf.Variable(tf.zeros(10))

    logits = tf.matmul(fc, fc_W) + fc_b

    return logits