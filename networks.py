import tensorflow as tf
from tensorflow.contrib.layers import flatten

def init_weights(shape, mean, stddev):
    return tf.Variable(tf.truncated_normal(shape = shape, mean = mean, stddev = stddev))

def init_biases(size):
    return tf.Variable(tf.zeros(size))

def lenet(x, mu = 0, sigma = 0.1):

    # Layer 1: Conv 32x32x1 => 28x28x6
    conv1_W = init_weights((5, 5, 1, 6), mu, sigma)
    conv1_b = init_biases(6)
    conv1 = tf.add(tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID'), conv1_b)
    conv1 = tf.nn.relu(conv1)
    # Layer 1 Pooling: 28x28x6 => 14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Conv 14x14x6 => 10x10x16
    conv2_W = init_weights((5, 5, 6, 16), mu, sigma)
    conv2_b = init_biases(16)
    conv2 = tf.add(tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID'), conv2_b)
    conv2 = tf.nn.relu(conv2)
    # Layer 2 Pooling: 10x10x16 => 5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten 5x5x16 => 400
    fc0 = flatten(conv2)

    # Layer 3: FC 400 => 120
    fc1_W = init_weights((400, 120), mu, sigma)
    fc1_b = init_biases(120)
    fc1 = tf.add(tf.matmul(fc0, fc1_W), fc1_b)
    fc1 = tf.nn.relu(fc1)

    # Layer 4: FC 120 => 84
    fc2_W = init_weights((120, 84), mu, sigma)
    fc2_b = init_biases(84)
    fc2 = tf.add(tf.matmul(fc1, fc2_W), fc2_b)
    fc2 = tf.nn.relu(fc2)

    # Layer 5: FC 84 => 10
    fc3_W = init_weights((84, 10), mu, sigma)
    fc3_b = init_biases(10)
    logits = tf.add(tf.matmul(fc2, fc3_W), fc3_b)

    return logits



