import matplotlib.pyplot as plt
from math import ceil
import numpy as np

colors = ["#037e87", "#00917f", "#31a065", "#73ab3f", "#b6ae0a", "#ffa600"] # Color palette used for visualization

def plot_images(images, cols = 1, labels = None):
    n_images = len(images)
    if labels is None: labels = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, label) in enumerate(zip(images, labels)):
        a = fig.add_subplot(np.ceil(n_images/float(cols)), cols, n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(label, fontsize=20)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def plot_image(image, gray=False):
    if gray:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

def pie(x, labels, title):
    plt.figure(figsize=(20,10))
    explode = (0.1, 0, 0)
    _, _, autotexts = plt.pie(x, startangle=90, autopct='%1.1f%%', explode=explode, labels=labels, shadow=True, colors=colors)
    for autotext in autotexts:
        autotext.set_color('white')
    plt.axis("equal")
    plt.title(title)
    plt.show()

def hist(x, bins, title):
    plt.figure(figsize=(20,10))
    _, _, patches = plt.hist(x, color=colors[1], rwidth=0.5, bins=np.arange(bins + 1), align="left")

    for index, p in enumerate(patches):
        plt.setp(p, 'facecolor', colors[index % len(colors)])
    plt.title(title)
    plt.xticks(np.arange(bins + 1))
    plt.show()