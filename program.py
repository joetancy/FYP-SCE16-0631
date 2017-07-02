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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(
    description='Image Retrieval')
parser.add_argument('-i1', '--input1', help='Input file name', required=True)
# parser.add_argument('-i2', '--input2', help='Input file name', required=True)
args = parser.parse_args()

with open("vgg16.tfmodel", mode='rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

images = tf.placeholder("float", [None, 224, 224, 3])

tf.import_graph_def(graph_def, input_map={"images": images})
print("Loaded tensorflow model")

graph = tf.get_default_graph()

image1 = utils.load_image(args.input1)
# image2 = utils.load_image(args.input2)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print("Model variables initialized")

    batch = image1.reshape((1, 224, 224, 3))
    assert batch.shape == (1, 224, 224, 3)

    feed_dict = {images: batch}
    vec_tensor = graph.get_tensor_by_name("import/fc8/BiasAdd:0")
    vector = sess.run(vec_tensor, feed_dict=feed_dict)
    location = np.squeeze(vector)


distance = sys.maxsize
nearestFile = None
i = 0
for root, dirs, files in os.walk("./vectors"):
    with progressbar.ProgressBar(max_value=len(files)) as bar:
        for file in files:
            if file.endswith(".vc"):
                imageLocation = np.loadtxt('./vectors/' + file, delimiter=',')
                dist = np.linalg.norm(location - imageLocation)
                if(dist < distance):
                    distance = dist
                    nearestFile = file
                i += 1
                bar.update(i)
print(distance)
print(nearestFile)
file = './images/' + nearestFile[:-3] + '.jpg'
img = Image.open('./images/' + nearestFile[:-3] + '.jpg')
img.show()
