import argparse
import os
import os.path
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.filters as filters
import skimage.util as utils

import imageUtilities

# parameter parser
parser = argparse.ArgumentParser(
    description='Image Distortion Thumbs')
parser.add_argument(
    '-i', '--image', help='Image location to distort', required=True)

args = parser.parse_args()


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def workerGaussian(image, distortParams):
    images = []
    for i in distortParams:
        image = filters.gaussian(
            image, sigma=i, truncate=2)
        images.append(image)
    show_images(images, cols=3, titles=distortParams)


def workerNoise(image, distortParams):
    images = []
    for i in distortParams:
        image = utils.random_noise(
            image, mode='gaussian', clip=True, var=distortParams[1])
        images.append(image)
    show_images(images, cols=3, titles=distortParams)


def workerJPEG(image, distortParams):
    images = []
    if not os.path.exists('./jpegthumbs'):
            os.makedirs('./jpegthumbs')
    for i in distortParams:
        skimage.io.imsave(
            './jpegthumbs/file' + str(i) + '.jpg', image, quality=int(i))
        images.append(imageUtilities.loadImage('./jpegthumbs/file'+str(i)+'.jpg', False))
    show_images(images, cols=3, titles=distortParams)
    shutil.rmtree('./jpegthumbs')


if __name__ == '__main__':
    image = imageUtilities.loadImage(args.image, False)
    distortParams = [3, 4, 5, 6, 7]
    workerGaussian(image, distortParams)
    distortParams = [0.01, 0.05, 0.10, 0.15, 0.20]
    workerNoise(image, distortParams)
    distortParams = [25, 20, 15, 10, 5]
    workerJPEG(image, distortParams)
