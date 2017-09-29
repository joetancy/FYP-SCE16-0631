import argparse
import os
import os.path
import sys
import time
import math

import ml_metrics as metrics
import numpy as np
import progressbar
import skimage.io
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

import imageUtilities

# reduce useless logs from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# parameter parser
parser = argparse.ArgumentParser(
    description='Image Retrieval')
parser.add_argument(
    '-i', '--image', help='Image location to search', required=True)

args = parser.parse_args()


def plotNNFilter(units):
    filters = 9
    ncol = math.ceil(math.sqrt(filters))
    nrow = ncol
    plt.figure(figsize=(ncol + 1, nrow + 1))
    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
    for i in range(filters):
        ax1 = plt.subplot(gs[i])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        plt.axis('off')
        plt.imshow(units[:, :, i], interpolation="nearest", cmap="gray")
    plt.show()


# opens the tf model of vgg16
with open("vgg16.tfmodel", mode='rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

images = tf.placeholder("float", [None, 224, 224, 3])

tf.import_graph_def(graph_def, input_map={"images": images})
print("Loaded VGG16 model")

graph = tf.get_default_graph()
image = imageUtilities.loadImage(args.image, False)

# for i in tf.get_default_graph().get_operations():
#     print(i.name)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print("Model variables initialized")

    batch = image.reshape((1, 224, 224, 3))
    assert batch.shape == (1, 224, 224, 3)

    feed_dict = {images: batch}
    vec_tensor = graph.get_tensor_by_name("import/conv1_1/Relu:0")
    vector = sess.run(vec_tensor, feed_dict=feed_dict)
    vector = np.squeeze(vector)
    plotNNFilter(vector)

    # vec_tensor = graph.get_tensor_by_name("import/conv1_2/Relu:0")
    # vector = sess.run(vec_tensor, feed_dict=feed_dict)
    # vector = np.squeeze(vector)
    # plotNNFilter(vector)

    # vec_tensor = graph.get_tensor_by_name("import/conv2_2/Relu:0")
    # vector = sess.run(vec_tensor, feed_dict=feed_dict)
    # vector = np.squeeze(vector)
    # plotNNFilter(vector)

    # vec_tensor = graph.get_tensor_by_name("import/conv3_3/Relu:0")
    # vector = sess.run(vec_tensor, feed_dict=feed_dict)
    # vector = np.squeeze(vector)
    # plotNNFilter(vector)

    # vec_tensor = graph.get_tensor_by_name("import/conv4_3/Relu:0")
    # vector = sess.run(vec_tensor, feed_dict=feed_dict)
    # vector = np.squeeze(vector)
    # plotNNFilter(vector)

    # vec_tensor = graph.get_tensor_by_name("import/conv5_3/Relu:0")
    # vector = sess.run(vec_tensor, feed_dict=feed_dict)
    # vector = np.squeeze(vector)
    # plotNNFilter(vector)
