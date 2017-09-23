import os
import os.path
import time

import numpy as np
import progressbar
import skimage
import skimage.filters as filters
import skimage.util as utils
from joblib import Parallel, delayed

import imageUtilities


def workerGaussian(file, root, imageDirList, distortParams):
    image = skimage.io.imread(os.path.join(root, file))
    image = skimage.util.img_as_float(image, force_copy=False)
    image = filters.gaussian(
        image, sigma=distortParams[0], truncate=2)
    skimage.io.imsave(
        './' + imageDirList[0] + '/' + file, image)


def workerNoise(file, root, imageDirList, distortParams):
    image = skimage.io.imread(os.path.join(root, file))
    image = skimage.util.img_as_float(image, force_copy=False)
    image = utils.random_noise(
        image, mode='gaussian', clip=True, var=distortParams[1])
    skimage.io.imsave(
        './' + imageDirList[1] + '/' + file, image)


def workerJPEG(file, root, imageDirList, distortParams):
    image = skimage.io.imread(os.path.join(root, file))
    image = skimage.util.img_as_float(image, force_copy=False)
    skimage.io.imsave(
        './' + imageDirList[2] + '/' + file, image, quality=distortParams[2])


def worker(file, root, imageDirList, distortParams):
    workerGaussian(file, root, imageDirList, distortParams)
    workerNoise(file, root, imageDirList, distortParams)
    workerJPEG(file, root, imageDirList, distortParams)


imageDirList = ['gaussian5', 'noise01', 'jpeg15']
distortParams = [5, 0.1, 15]

if __name__ == '__main__':

    for i in imageDirList:
        if not os.path.exists(i):
            os.makedirs(i)

    i = 0

    for root, dirs, files in os.walk('./holidays'):
        Parallel(n_jobs=-1, verbose=5)(delayed(worker)(
            file, root, imageDirList, distortParams) for file in files)
