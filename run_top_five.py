import pickle
import numpy as np
from nn import NN
from image import pre_process, augment
import tensorflow as tf
import itertools
import time
from sklearn.utils import shuffle 
import pandas as pd

""" training_file = "data/train.p"
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
aug_X = pre_process(aug_X) """

experiments = pd.read_csv('experiment_stats.csv')


def explode_accuracy(s):
    s['validation_accuracy'] = s['validation_accuracy'][1:-2].split(', ')[-1]
    return s

experiments = experiments.apply(explode_accuracy, axis=1)

experiments = experiments.sort_values(by=['validation_accuracy'], ascending=False)

top_5 = experiments[:5]