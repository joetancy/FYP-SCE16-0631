import imageUtilities
import numpy as np
import os
import time
import progressbar
import os.path
import skimage
import skimage.filters as filters
import skimage.util as utils

imageDir = input('Image directory: ')
saveImageDir = input('Save image directory: ')

distortionType = input(
    '1. Gaussian Blur\n2. Gaussian Noise\n3. JPEG Compression\nEnter your choice: ')

if distortionType == '1':
    print('Gaussian Blur\n')
    # gaussian blur
    i = 0
    for root, dirs, files in os.walk("./" + imageDir):
        with progressbar.ProgressBar(max_value=len(files)) as bar:
            for file in files:
                if file.endswith(".jpg"):
                    print('\nDistorting for', file)
                    image = skimage.io.imread(os.path.join(root, file))
                    image = skimage.util.img_as_float(image, force_copy=False)
                    image = filters.gaussian(image, sigma=3, truncate=2)
                    time.sleep(0.1)
                    file = './' + saveImageDir + '/' + file
                    skimage.io.imsave(file, image)
                i += 1
                bar.update(i)
elif distortionType == '2':
    print('Gaussian Noise\n')
    # gaussian noise
    i = 0
    for root, dirs, files in os.walk("./" + imageDir):
        with progressbar.ProgressBar(max_value=len(files)) as bar:
            for file in files:
                if file.endswith(".jpg"):
                    print('\nDistorting for', file)
                    image = skimage.io.imread(os.path.join(root, file))
                    image = skimage.util.img_as_float(image, force_copy=False)
                    image = utils.random_noise(
                        image, mode='gaussian', clip=True)
                    time.sleep(0.1)
                    file = './' + saveImageDir + '/' + file
                    skimage.io.imsave(file, image)
                i += 1
                bar.update(i)
elif distortionType == '3':
    print('JPEG Compression\n')
    # jpeg compression
    i = 0
    for root, dirs, files in os.walk("./" + imageDir):
        with progressbar.ProgressBar(max_value=len(files)) as bar:
            for file in files:
                if file.endswith(".jpg"):
                    print('\nDistorting for', file)
                    image = skimage.io.imread(os.path.join(root, file))
                    image = skimage.util.img_as_float(image, force_copy=False)
                    time.sleep(0.1)
                    file = './' + saveImageDir + '/' + file
                    skimage.io.imsave(file, image, quality=25)
                i += 1
                bar.update(i)
