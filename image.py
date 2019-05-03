import cv2
import numpy as np

def explore(images):
    pass

def pre_process(images):
    return gray(standardize(images))

def augment(images, factor):
    pass

def gray(images):
    return [np.sum(image/3, axis=2, keepdims=True) for image in images]

def standardize(images):
    return [(image - np.mean(image)) / np.std(image) for image in images]
""" 
def hsv(images):
    return tf.image.rgb_to_hsv(images)

def yiq(images):
    return tf.image.rgb_to_yiq(images)

def yuv(images):
    return tf.image.rgb_to_yuv(images)

def rotate(images, n):
    return tf.image.rot90(images.images, k=n)

def sobel(images):
    return tf.image.sobel_edges(images) """
