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
from models.vgg19 import build_model


def prep_image(url, mean_image):
    
    ''' Crops and centers an ImageNet image to 3 x 224 x 224. 
    
        Return: 
        :rawim: the cropped image without mean-centering
        :im: a mean-centered image 
    '''
    
    ext = url.split('.')[-1]
    im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)
    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]

    rawim = np.copy(im).astype('uint8')

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - mean_image[:, None, None]
    return rawim, floatX(im[np.newaxis])


def get_images(num_images=5):
    index = urllib.urlopen('http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html').read()
    image_urls = index.split('<br>')

    np.random.seed(23)
    np.random.shuffle(image_urls)
    image_urls = image_urls[:num_images]
    return image_urls


def load_data():
    with open("vgg19.pkl", "r") as f:
        model = pickle.load(f)

    classes = model["synset words"]
    mean_image = model["mean value"]
    values = model["param values"]

    return classes, mean_image, values


if __name__ == "__main__":
    classes, mean_image, values = load_data()

    net = build_model()
    output_layer = net["prob"]
    lasagne.layers.set_all_param_values(output_layer, values)

    image_urls = get_images(3)


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
