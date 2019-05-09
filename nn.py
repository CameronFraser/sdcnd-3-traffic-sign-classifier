import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
from sklearn.utils import shuffle

class NN:
    def __init__(self, epochs=10, batch_size=128, learning_rate=0.001, name="nn"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.accuracy_history = []
        self.layers = []

    def add_train_data(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def add_validation_data(self, X_valid, y_valid):
        self.X_valid = X_valid
        self.y_valid = y_valid

    def add_test_data(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def add_configuration(self, config, input_size):
        self.config = config
        height, width, current_input = input_size
        for layer in config:
            if layer['type'] == 'conv':
                self.add_layer(self.conv_layer(current_input, layer['filters'], layer['ksize'], layer['stride']))
                current_input = layer['filters']
            elif layer['type'] == 'max_pool':
                self.add_layer(self.max_pool(layer['ksize'], layer['stride']))
                height //= layer['stride'][0]
                width //= layer['stride'][1]
            elif layer['type'] == 'flatten':
                self.add_layer(self.flatten())
                current_input = (current_input * height * width)
            elif layer['type'] == 'dropout':
                self.add_layer(self.dropout())
            elif layer['type'] == 'fc':
                self.add_layer(self.fc_layer(current_input, layer['units']))
                current_input = layer['units']
            elif layer['type'] == 'relu':
                self.add_layer(self.relu())
    
    def build(self, num_labels):
        self.x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        self.y = tf.placeholder(tf.int32, (None))
        self.one_hot_y = tf.one_hot(self.y, num_labels)
        self.keep_prob = tf.placeholder(tf.float32)
        
        local_x = self.x
        for layer in self.layers:
            local_x = layer(local_x)
        
        self.saver = tf.train.Saver()
        self.logits = local_x
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y, logits=self.logits)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.training_operation = self.optimizer.minimize(self.loss_operation)
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    

    def train(self, keep_prob = 0.5):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(self.X_train)
            
            print("Training...")
            for i in range(self.epochs):
                self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
                for offset in range(0, num_examples, self.batch_size):
                    end = offset + self.batch_size
                    batch_x, batch_y = self.X_train[offset:end], self.y_train[offset:end]
                    sess.run(self.training_operation, feed_dict={ self.x: batch_x, self.y: batch_y, self.keep_prob: keep_prob})
                    
                validation_accuracy = self.evaluate(self.X_valid, self.y_valid)
                self.accuracy_history.append(validation_accuracy)
                print("EPOCH {}".format(i+1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()
            self.saver.save(sess, './savednetwork')
        return self.accuracy_history

    def evaluate(self, X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, self.batch_size):
            batch_x, batch_y = X_data[offset:offset+self.batch_size], y_data[offset:offset+self.batch_size]
            accuracy = sess.run(self.accuracy_operation, feed_dict={ self.x: batch_x, self.y: batch_y, self.keep_prob: 1.0} )
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def test(self):
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('.'))

            test_accuracy = self.evaluate(self.X_test, self.y_test)
            print("Test Accuracy = {:.3f}".format(test_accuracy))
            return test_accuracy


    def init_weights(self, shape):
        mu = 0
        sigma = 0.1
        return tf.Variable(tf.truncated_normal(shape = shape, mean = mu, stddev = sigma))
    
    def init_biases(self, size):
        return tf.Variable(tf.zeros(size))
    
    def relu(self):
        return lambda x: tf.nn.relu(x)

    def conv_layer(self, size_in, size_out, ksize, strides, name="conv"):
        W = self.init_weights(ksize + [size_in, size_out])
        b = self.init_biases(size_out)
        return lambda x: tf.add(tf.nn.conv2d(x, W, [1] + strides + [1], padding="SAME"), b)

    def fc_layer(self, size_in, size_out, name="fc"):
        W = self.init_weights([size_in, size_out])
        b = self.init_biases(size_out)
        return lambda x: tf.add(tf.matmul(x, W), b)
    
    def max_pool(self, ksize, strides, name="max_pool"):
        return lambda x: tf.nn.max_pool(x, ksize=[1] + ksize + [1], strides=[1] + strides + [1], padding="SAME")

    def flatten(self):
        return lambda x: flatten(x)

    def dropout(self):
        return lambda x: tf.nn.dropout(x, self.keep_prob)

    def get_string(self):
        nn_layers = []
        for layer in self.config:
            nn_layers.append(layer['type'])
        return ' --> '.join(nn_layers)
    