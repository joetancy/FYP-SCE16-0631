import tensorflow as tf
import imageUtilities
import numpy as np
import os
import time
import progressbar
import os.path
from imageDistortion import imageDirList

# reduce useless logs from tensorflow
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

imageDirList = imageDirList


with open("vgg16.tfmodel", mode='rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

images = tf.placeholder("float", [None, 224, 224, 3])

tf.import_graph_def(graph_def, input_map={"images": images})
print("Loaded tensorflow model")
graph = tf.get_default_graph()

for directories in imageDirList:
    print(directories)
    imageDir = directories
    featuresDir = imageDir + 'Features'
    if not os.path.exists(featuresDir):
        os.makedirs(featuresDir)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("Model variables initialized")
        i = 0

        for root, dirs, files in os.walk('./' + imageDir):
            with progressbar.ProgressBar(max_value=len(files)) as bar:
                for file in files:
                    if file.endswith(".jpg"):
                        if (not (os.path.exists(os.path.join('./' + featuresDir + '/' + file[:-4] + '.vc')))):
                            i += 1
                            image = imageUtilities.loadImage(
                                os.path.join(root, file), False)
                            batch = image.reshape((1, 224, 224, 3))
                            assert batch.shape == (1, 224, 224, 3)
                            feed_dict = {images: batch}
                            vec_tensor = graph.get_tensor_by_name(
                                "import/fc7/BiasAdd:0")
                            vector = sess.run(vec_tensor, feed_dict=feed_dict)
                            location = np.squeeze(vector)
                            file = file[:-4]
                            file = './' + featuresDir + '/' + file + '.vc'
                            np.savetxt(file, location,
                                       delimiter=',', fmt="%s")
                        else:
                            i += 1
                        bar.update(i)
