import pickle
import numpy as np
from nn import NN
from image import pre_process
import tensorflow as tf
import itertools
from collections import OrderedDict 

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

top_5 = []

# Hyperparameters we will iterate over for experiments
data_augmentation = [True, False]
learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
batch_sizes = [32, 64, 128, 256, 512]
dropout_keep_prob = [0.3, 0.4, 0.5, 0.6, 0.7]

network_configurations = [
    [
        { 'type': 'conv', 'filters': 32, 'ksize': [5, 5], 'stride': [1, 1] },
        { 'type': 'max_pool', 'ksize': [2, 2], 'stride': [2, 2] },
        { 'type': 'conv', 'filters': 64, 'ksize': [5, 5], 'stride': [1, 1] },
        { 'type': 'max_pool', 'ksize': [2, 2], 'stride': [2, 2] },
        { 'type': 'flatten' },
        { 'type': 'fc', 'units': 1024 },
        { 'type': 'dropout' },
        { 'type': 'fc', 'units': 400 },
        { 'type': 'dropout' },
        { 'type': 'fc', 'units': 43 }
    ],
    [
        { 'type': 'conv', 'filters': 16, 'ksize': [5, 5] },
        { 'type': 'max_pool', 'ksize': [2, 2], 'stride': [2, 2] },
        { 'type': 'flatten' },
        { 'type': 'fc', 'units': 84 },
        { 'type': 'dropout' },
        { 'type': 'fc', 'units': 43 }
    ]
]

hyperparameters = [data_augmentation, learning_rates, batch_sizes, dropout_keep_prob, network_configurations]

experiments = list(itertools.product(*hyperparameters))

print("{} experiments about to run.".format(len(experiments)))

network = NN(epochs=10, batch_size=128)
network.add_train_data(X_train, y_train)
network.add_test_data(X_test, y_test)
network.add_validation_data(X_valid, y_valid)

network.add_configuration(network_configurations[0], input_size=X_train[0].shape)

network.build(num_labels=max(y_train) + 1)

network.train()

