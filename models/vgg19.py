# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import urllib
import io
import skimage.transform

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.conv import Conv2DLayer as ConvLayer
from lasagne.layers.pool import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from lasagne.nonlinearities import softmax

def prep_image(url, mean_image):
    ext = url.split('.')[-1]
    try:
        im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)

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
        im = im - mean_image[:,None,None]
        return rawim, floatX(im[np.newaxis])

    except:
        # Abort
        print "skipping url " + url
        return None, np.zeros((1,))
    

# Model to be trained/tested against
def build_model():
    """ 
    Builds the classic VGG-19 model using Lasagne wrapper to Theano.
    """
    
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1)
    net['conv3_4'] = ConvLayer(
        net['conv3_3'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_4'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1)
    net['conv4_4'] = ConvLayer(
        net['conv4_3'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_4'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1)
    net['conv5_4'] = ConvLayer(
        net['conv5_3'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_4'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net

def load_weights():
    with open("../weights/vgg19.pkl", "r") as f:
        model = pickle.load(f)

    classes = model["synset words"]
    mean_image = model["mean value"]
    values = model["param values"]

    return classes, mean_image, values

# Validation Set
def download_val_images (num_ex, mean_image):
    ''' 
    Dynamically downloads sample images from ImageNet.  
    '''
    index = urllib.urlopen('http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html').read()
    image_urls = index.split('<br>')

    np.random.seed(19)
    np.random.shuffle(image_urls)

    images = np.zeros ((num_ex, 3, 224, 224), dtype=np.float32)
    i = 0
    used=True
    for im_url in image_urls:
        print i, im_url
        # only use quick downloads on flickr
        if 'static.flickr' not in im_url:
            continue
            
        _, result = prep_image (im_url, mean_image)
      
        if result.any():
            images[i,:,:,:] = result
            i += 1
           
        if i >= num_ex: 
            break
            
    print images.shape
    return images

# Here you can save the images to a pickled file.
def save_images (images, dataset="imagenet_val"):
    """ 
    Saves an array of images to file.
    
    """
    filename = dataset + "_" + str(images.shape[0])
    with open('../datasets/'+filename+'.pkl', 'w+') as f:
        pickle.dump(images, f)

def load_images (filename):
    """ 
    Loads images from file and returns it.
    """
    data = None
    with open('../datasets/' + filename, 'rb') as f:
        data = pickle.load(f)
    return data

from time import time
# Process test images and print top 5 labels
def run_forward(images):

    print images.shape
    prob = np.array(lasagne.layers.get_output(net['prob'], images, deterministic=True).eval())

    
