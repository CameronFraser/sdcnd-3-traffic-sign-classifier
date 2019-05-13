import pickle
import numpy as np
from nn import NN
from image import pre_process, augment
import tensorflow as tf
import itertools
import time
from sklearn.utils import shuffle 
import pandas as pd
import cv2

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

aug_X, aug_y = augment(X_train, y_train)
aug_X, aug_y = shuffle(aug_X, aug_y)

X_train = pre_process(X_train)
X_valid = pre_process(X_valid)
X_test = pre_process(X_test)
aug_X = pre_process(aug_X)


# Hyperparameters we will iterate over for experiments
data_augmentation = [str(True), str(False)]
batch_sizes = [32, 64, 128]
dropout_keep_prob = [0.4, 0.5, 0.6]
config = [0, 1]
filters = [[16, 32], [32, 64], [64, 128]]
ksize = [[1, 1], [2, 2], [3, 3], [4, 4]]

def create_network_config(index, filters, ksize):
    network_configurations = [
        [
            { 'type': 'conv', 'filters': filters[0], 'ksize': ksize, 'stride': [1, 1] },
            { 'type': 'max_pool', 'ksize': [2, 2], 'stride': [2, 2] },
            { 'type': 'relu' },
            { 'type': 'conv', 'filters': filters[1], 'ksize': ksize, 'stride': [1, 1] },
            { 'type': 'max_pool', 'ksize': [2, 2], 'stride': [2, 2] },
            { 'type': 'relu' },
            { 'type': 'flatten' },
            { 'type': 'fc', 'units': 128 },
            { 'type': 'dropout' },
            { 'type': 'relu' },
            { 'type': 'fc', 'units': 43 }
        ],
        [
            { 'type': 'conv', 'filters': filters[0], 'ksize': ksize, 'stride': [1, 1] },
            { 'type': 'relu' },
            { 'type': 'max_pool', 'ksize': [2, 2], 'stride': [2, 2] },
            { 'type': 'flatten' },
            { 'type': 'fc', 'units': 128 },
            { 'type': 'dropout' },
            { 'type': 'relu' },
            { 'type': 'fc', 'units': 43 }
        ]
    ]
    return network_configurations[index]

hyperparameters = [data_augmentation, batch_sizes, dropout_keep_prob, filters, ksize, config]

# Get every permutation of hyperparameters
experiments = list(itertools.product(*hyperparameters))



input_size = X_train[0].shape
num_labels = max(y_train) + 1

# stats will contain the statistics from the experiments that are ran
# hyperparameters, total time, accuracy
stats = pd.read_csv('experiment_stats.csv')
stat_labels = [
    'elapsed_time_to_train',
    'validation_accuracy',
    'data_augmented',
    'batch_size',
    'dropout_keep_probability',
    'conv_filters',
    'conv_ksize',
    'architecture'
]

tf.logging.set_verbosity(tf.logging.ERROR)

# seeding the shuffle in case the computer crashes and we need to restart from where we left off
experiments = shuffle(experiments, random_state=99)

experiments = experiments[400:]
print("{} experiments about to run.".format(len(experiments)))

for experiment in experiments:
    augment, batch_size, keep_prob, filters, ksize, config = experiment
    start_time = time.time()
    network = NN(epochs=5, batch_size=batch_size, learning_rate=0.001)
    features = np.concatenate([X_train, aug_X]) if augment == 'True' else X_train
    labels = np.concatenate([y_train, aug_y]) if augment == 'True' else y_train
    network.add_train_data(features, labels)
    network.add_test_data(X_test, y_test)
    network.add_validation_data(X_valid, y_valid)

    network.add_configuration(create_network_config(config, filters, ksize), input_size=input_size)

    network.build(num_labels=num_labels)
    print("Training model with hyperparameters: augmented: {}, batch_size: {}, keep_prob: {}, filters: {}, ksize: {}, config: {}".format(augment, batch_size, keep_prob, filters, ksize, network.get_string()))
    validation_accuracy = network.train(keep_prob=keep_prob)

    end_time = time.time()

    stat_values = [end_time - start_time, validation_accuracy, augment, batch_size, keep_prob, filters, ksize, network.get_string()]
    stat_entry = pd.Series(stat_values, index=stat_labels)
    stats = stats.append(stat_entry, ignore_index=True)
    stats.to_csv('experiment_stats.csv', index = None, header=True)
