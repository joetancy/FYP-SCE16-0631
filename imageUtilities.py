import skimage
import skimage.io
import skimage.transform
import sys
import numpy as np
from operator import itemgetter
from PIL import Image


def loadImage(path, display):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(
        crop_img, (224, 224), mode='constant')
    if display:
        skimage.io.imshow(resized_img)
        skimage.io.show()
    return resized_img


def initList(n):
    max_size_list = []
    for i in range(n):
        max_size_list.append({'filename': 'null', 'distance': sys.maxsize})
    return max_size_list


def addToNeighbours(neighbour, neighbour_list):
    neighbour_list.append(neighbour)
    newlist = sorted(neighbour_list, key=itemgetter('distance'))
    newlist.pop()
    return newlist


def getMaxNeighbour(neighbour_list):
    return neighbour_list[-1]['distance']


def prettyPrintList(neighbour_list):
    for i in neighbour_list:
        print('\n filename: ', i['filename'], ' distance: ', i['distance'])


def openAllImages(neighbour_list):
    for i in neighbour_list:
        file = './images/' + i['filename']
        skimage.io.imshow(file)
        skimage.io.show()
