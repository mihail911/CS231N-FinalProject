import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ubuntu/.local/lib/python2.7/site-packages/')
import pickle
import urllib
import io
import skimage.transform
#Bfrom gen_adver import FastGradient
from textwrap import wrap
import os


import multiprocessing as mp
    
# import theano
# import theano.tensor as T
# from theano import pp
# import time

# import lasagne
# from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
# from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
# from lasagne.layers import NonlinearityLayer
# from lasagne.layers import DropoutLayer
# from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
# from lasagne.utils import floatX
#from lasagne.nonlinearities import softmax

import socket

class DiskReader (object):
    ''' 
    Provides images off of disk over a socket, and stores
    data locally with a synset.
    
    One thread servers as the base server to bind to, and the other does
    computation in the background.
    '''
    def __init__(self):
        # map synset id : (semaphore, marshall-ready numpy 4D preprocessed image array)
        self.data = {}
        self.
        self.activeQueue = 

    def startRequest (self, synset):
        
        self.activeQueue = mp.Queue()
        proc = mp.Process(target=processImages, args=((synset, self.activeQueue) ))
        proc.start()
        

    def processImages (self, args):
        synset, q = args
        mean_image= np.array([119., 126., 131.])):
        ''' Processes the images from a directory on disk '''
        count = 0
        prefix = '../datasets/{0}/'.format(synset)
        files = os.listdir(prefix)
        N = len(files)
        images = np.zeros ((N, 3, 224, 224))

        for i, f in enumerate(files):
            im = plt.imread (prefix + f)
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
            count += 1
            if count % 10 == 0:
                print count    
            # Convert to BGR
        all_images = im[::-1, :, :]

        all_images -= mean_image[None,:,None,None]
        q.put( floatX(all_images[np.newaxis]) )

''' 
Implements bootstrapping on a class.
'''
class Bootstrap(object):
    
    def __init__(self, synset, network=None):
        '''
        Params:
        - synset: the class to boostrap off of
        - network: the lasagne / theano DNN to use
        '''
        self.synset = synset
        self.network = network

if __name__ == '__main__':
    bts = Bootstrap('n01440764')
    bts.processImages()

