import tensorflow as tf
import utils
import numpy as np
import argparse
import os
import os.path
import sys
import time
import progressbar
from PIL import Image

# reduce useless logs from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# nearest neighbour
n = 3

# parameter parser
parser = argparse.ArgumentParser(
    description='Image Retrieval')
parser.add_argument(
    '-i', '--image', help='Image location to search', required=True)

args = parser.parse_args()

# opens the tf model of vgg16
with open("vgg16.tfmodel", mode='rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

images = tf.placeholder("float", [None, 224, 224, 3])

tf.import_graph_def(graph_def, input_map={"images": images})
print("Loaded VGG16 model")

graph = tf.get_default_graph()
image = utils.loadImage(args.image)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print("Model variables initialized")

    batch = image.reshape((1, 224, 224, 3))
    assert batch.shape == (1, 224, 224, 3)

    feed_dict = {images: batch}
    # second last layer of the model before last activation
    vec_tensor = graph.get_tensor_by_name("import/fc8/BiasAdd:0")
    vector = sess.run(vec_tensor, feed_dict=feed_dict)
    # location of input image
    location = np.squeeze(vector)

nearestNeighbours = utils.initList(n)

i = 0
# Calculating input image location with all other location of the ukbench library
print("Calculating distance vectors...")
for root, dirs, files in os.walk("./vectors"):
    with progressbar.ProgressBar(max_value=len(files)) as bar:
        for file in files:
            if file.endswith(".vc"):
                imageLocation = np.loadtxt('./vectors/' + file, delimiter=',')
                dist = np.linalg.norm(location - imageLocation)
                if(dist < utils.getMaxNeighbour(neighbour_list=nearestNeighbours)):
                    nearestNeighbours = utils.addToNeighbours(
                        neighbour={'filename': file[:-3] + '.jpg', 'distance': dist}, neighbour_list=nearestNeighbours)
                i += 1
                bar.update(i)

print('Showing', n, 'closest images')
utils.prettyPrintList(nearestNeighbours)

# file = './images/' + nearestFile[:-3] + '.jpg'
# img = Image.open('./images/' + nearestFile[:-3] + '.jpg')
# img.show()
