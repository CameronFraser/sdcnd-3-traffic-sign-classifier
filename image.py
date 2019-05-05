import cv2
import numpy as np

def pre_process(images):
    return gray(standardize(images))

def augment(images, factor):
    pass

def gray(images):
    return [np.sum(image/3, axis=2, keepdims=True) for image in images]

def standardize(images):
    return [(image - np.mean(image)) / np.std(image) for image in images]


