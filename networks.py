import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np

def init_weights(shape):
    mu = 0
    sigma = 0.1
    return tf.Variable(tf.truncated_normal(shape = shape, mean = mu, stddev = sigma))

def init_biases(size):
    return tf.Variable(tf.zeros(size))

def conv_layer(x, size_in, size_out, strides=[1, 1, 1, 1], name="conv"):
    W = init_weights([5, 5, size_in, size_out])
    b = init_biases(size_out)
    conv = tf.nn.conv2d(x, W, strides, padding="SAME")
    conv = tf.add(conv, b)
    return tf.nn.relu(conv)

def fc_layer(x, size_in, size_out, name="fc"):
    W = init_weights([size_in, size_out])
    b = init_biases(size_out)
    fc = tf.add(tf.matmul(x, W), b)
    return tf.nn.relu(fc)

def nn(x, keep_prob):
    # Layer 1: Conv 32x32x1 => 32x32x32
    conv1 = conv_layer(x, 1, 32)
    # Layer 1 Pooling: 32x32x32 => 16x16x32
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Layer 2: Conv 16x16x32 => 16x16x64
    conv2 = conv_layer(conv1, 32, 64)
    # Layer 2 Pooling: 16x16x64 => 8x8x64
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Flatten 8x8x64 => 4096
    fc0 = flatten(conv2)

    # Layer 3: FC 4096 => 2048
    fc1 = tf.nn.dropout(fc_layer(fc0, 4096, 1024), keep_prob)

    # Layer 4: FC 2048 =>
    fc2 = tf.nn.dropout(fc_layer(fc1, 1024, 400), keep_prob)

    logits = tf.nn.dropout(fc_layer(fc2, 400, 43), keep_prob)


    return logits



