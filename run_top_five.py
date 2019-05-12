import pickle
import numpy as np
from nn import NN
from image import pre_process, augment
import tensorflow as tf
import itertools
import time
from sklearn.utils import shuffle 
import pandas as pd

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

aug_X, aug_y = augment(X_train, y_train)
aug_X, aug_y = shuffle(aug_X, aug_y)
aug_X = pre_process(aug_X)

input_size = X_train[0].shape
num_labels = max(y_train) + 1

experiments = pd.read_csv('experiment_stats.csv')

def explode_accuracy(s):
    s['validation_accuracy'] = s['validation_accuracy'][1:-2].split(', ')[-1]
    return s

experiments = experiments.apply(explode_accuracy, axis=1)

experiments = experiments.sort_values(by=['validation_accuracy'], ascending=False)

top_5 = experiments[:5]

stats = None
stat_labels = [
    'elapsed_time_to_train',
    'validation_accuracy',
    'test_accuracy',
    'data_augmented',
    'learning_rate',
    'batch_size',
    'dropout_keep_probability',
    'architecture'
]

tf.logging.set_verbosity(tf.logging.ERROR)

for idx, row in top_5.iterrows():
    architecture_name, batch_size, augment, keep_prob, _, rate, _ = row
    architecture = [
        { 'type': 'conv', 'filters': 32, 'ksize': [3, 3], 'stride': [1, 1] },
        { 'type': 'max_pool', 'ksize': [2, 2], 'stride': [2, 2] },
        { 'type': 'relu' },
        { 'type': 'conv', 'filters': 64, 'ksize': [3, 3], 'stride': [1, 1] },
        { 'type': 'max_pool', 'ksize': [2, 2], 'stride': [2, 2] },
        { 'type': 'relu' },
        { 'type': 'flatten' },
        { 'type': 'fc', 'units': 128 },
        { 'type': 'dropout' },
        { 'type': 'relu' },
        { 'type': 'fc', 'units': 43 }
    ]
    start_time = time.time()
    network = NN(epochs=50, batch_size=int(batch_size), learning_rate=rate)
    features = np.concatenate([X_train, aug_X]) if augment == 'True' else X_train
    labels = np.concatenate([y_train, aug_y]) if augment == 'True' else y_train
    network.add_train_data(features, labels)
    network.add_test_data(X_test, y_test)
    network.add_validation_data(X_valid, y_valid)

    network.add_configuration(architecture, input_size=input_size)

    network.build(num_labels=num_labels)
    print("Training model with hyperparameters: augmented: {}, rate: {}, batch_size: {}, keep_prob: {}, config: {}".format(augment, rate, batch_size, keep_prob, architecture_name))
    validation_accuracy = network.train(keep_prob=keep_prob, save_name="test" + str(idx))

    end_time = time.time()

    test_accuracy = network.test(save_name="test" + str(idx))

    try:
        stats = pd.read_csv('top_5_stats.csv')
    except:
        stats = pd.DataFrame()
    
    stat_values = [end_time - start_time, validation_accuracy, test_accuracy, augment, rate, batch_size, keep_prob, architecture_name]
    stat_entry = pd.Series(stat_values, index=stat_labels)
    stats = stats.append(stat_entry, ignore_index=True)
    stats.to_csv('top_5_stats.csv', index = None, header=True)

    