import cv2
import numpy as np

def pre_process(images):
    return gray(standardize(images))
             
def rotate(images, factor=3):
    one_transform = [np.rot90(img, k=1) for img in images]
    two_transform = [np.rot90(img, k=2) for img in images]
    three_transform = [np.rot90(img, k=3) for img in images]
    return np.concatenate([one_transform, two_transform, three_transform])

def flip(images, factor=3):
    one_transform = [np.fliplr(img) for img in images]
    two_transform = [np.flipud(img) for img in images]
    three_transform = [np.flipud(np.fliplr(img)) for img in images]
    return np.concatenate([one_transform, two_transform, three_transform])

def roll(images, factor=3):
    one_transform = [np.roll(img, 10) for img in images]
    two_transform = [np.roll(img, 10, axis=0) for img in images]
    three_transform = [np.roll(img, 10, axis=1) for img in images]
    return np.concatenate([one_transform, two_transform, three_transform])

def get_factors(images, labels):
    label_freq = {}
    categorized_images = {}

    for idx, label in enumerate(labels):
        if label in label_freq:
            label_freq[label] += 1
        else:
            label_freq[label] = 1
        if label in categorized_images:
            categorized_images[label].append(images[idx])
        else:
            categorized_images[label] = [images[idx]]
    
    freq_list = list(label_freq.values())
    max_freq = max(freq_list)
    return [round(max_freq / freq) for freq in freq_list]


def augment(images, labels):
    pipeline = [rotate, flip, roll]

    factors = get_factors(images, labels)
    augmented_X = None
    augmented_y = []
    for idx, factor in enumerate(factors):
        if factor > 1:
            calls = factor // 3
            for call_idx in range(calls):
                try:
                    filtered_X = []
                    for i, X in enumerate(images):
                        if labels[i] == idx:
                            filtered_X.append(X)
                    new_X = pipeline[call_idx](filtered_X)
                    
                    if augmented_X == None:
                        augmented_X = new_X
                    else:
                        augmented_X = np.concatenate([augmented_X, new_X])
                    
                    augmented_y = np.concatenate([augmented_y, np.repeat(idx, len(new_X))])
                except IndexError:
                    pass

    return augmented_X, augmented_y



def gray(images):
    return [np.sum(image/3, axis=2, keepdims=True) for image in images]

def standardize(images):
    return [(image - np.mean(image)) / np.std(image) for image in images]


