import cPickle as pickle
import io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# Janky work around for making sure the module can find 'Lasagne'
sys.path.append("/Users/mihaileric/Documents/Research/Lasagne")

import urllib


import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.conv import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from models import vgg19

if __name__ == "__main__":
    classes, mean_image, values = vgg19.load_data()

    net = vgg19.build_model()
    output_layer = net["prob"]
    lasagne.layers.set_all_param_values(output_layer, values)

    image_urls = vgg19.get_images(3)

    count = 0
    for url in image_urls:
        print url
        try:
            rawim, im = prep_image(url, mean_image)

            prob = np.array(lasagne.layers.get_output(output_layer, im, deterministic=True).eval())
            top5 = np.argsort(prob[0])[-1:-6:-1]
            fig = plt.figure()
            plt.imshow(rawim.astype('uint8'))
            plt.axis('off')
            for n, label in enumerate(top5):
                plt.text(250, 70 + n * 20, '{}. {}'.format(n+1, classes[label]), fontsize=14)
            plt.savefig("Image:"+str(count))
            plt.clf()
            plt.close(fig)
            count += 1
        except IOError:
            print('bad url: ' + url)
