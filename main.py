import pickle
import numpy as np
from nn import NN
from image import pre_process
import tensorflow as tf

training_file = "data/train.p"
validation_file = "data/valid.p"
testing_file = "data/test.p"


with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

X_train = pre_process(X_train)
X_valid = pre_process(X_valid)
X_test = pre_process(X_test)

network = NN()
network.add_train_data(X_train, y_train)
network.add_test_data(X_test, y_test)
network.add_validation_data(X_valid, y_valid)

network.add_layers([
    network.conv_layer(1, 32),
    network.max_pool(),
    network.conv_layer(32, 64),
    network.max_pool(),
    network.flatten(),
    network.fc_layer(4096, 1024),
    network.dropout(),
    network.fc_layer(1024, 400),
    network.dropout(),
    network.fc_layer(400, 43)
])

network.build(num_labels=43)

network.train()

""" Training...
EPOCH 1
Validation Accuracy = 0.663

EPOCH 2
Validation Accuracy = 0.858

EPOCH 3
Validation Accuracy = 0.897

EPOCH 4
Validation Accuracy = 0.925

EPOCH 5
Validation Accuracy = 0.930

EPOCH 6
Validation Accuracy = 0.934

EPOCH 7
Validation Accuracy = 0.931

EPOCH 8
Validation Accuracy = 0.953

EPOCH 9
Validation Accuracy = 0.950

EPOCH 10
Validation Accuracy = 0.948 """

