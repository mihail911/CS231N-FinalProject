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
#from gen_adver import FastGradient
from textwrap import wrap
import os
import sys
import shutil

import multiprocessing as mp
    
import theano
import theano.tensor as T
from theano import pp
import time

from sklearn.decomposition import PCA

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX

# local
from diskreader import DiskReader

''' 
Implements bootstrapping on a class.
'''
class Bootstrap(object):
    
    def __init__(self, network = None, diskReader = None):
        self.network = network
        self.reader = diskReader if diskReader is not None else DiskReader()


    def forward(self, synset, nextsyn=None):

        ''' 
        Takes a synset and loads the necessary data from disk.  
        Returns activations from the 4096 feature layer for the given synset

        If given nextsyn, will request to load it up.
        ''' 

        images = self.reader.get(synset)
        if nextsyn:
	    self.reader.startRequest(nextsyn)
	print images.shape
        N, C, H, W = images.shape
        features = np.zeros ((N, 4096))

        for n in xrange(0, N, batch_size):
	    t0 = time.time()
            curSet = images[n:n+batch_size, :, :, :]
	    print "beginning run on {0} : {1}".format(n, n+batch_size)
            features[n:n+batch_size, :] = np.array(lasagne.layers.get_output(
                                                    net['fc6'], curSet, deterministic=True).eval(), 
                                                    dtype=np.float32)
	    t1 = time.time()
	    print "batch {0} - {1} : took {2:.4f} seconds".format(n, n+batch_size, t1 - t0)
        
	# remove the imags from the reader, since we don't need them anymore.
	self.reader.remove(synset) 
	return images, features


    def sample (self, synset, features, method='l2', k=50, num_samples=3000, pc=128):
        '''  
        Bootstraps on the input space using the given method.

	Does an RBF kernel PCA transform.
        Params:
        - features : a matrix of feature vectors, with the first dimension N.  

        Returns:
        - samples: the set of bootstrapped size-k samples
        '''
	print features.shape        
        N = features.shape[0]
	if N == 0:
	    return None
        mean_img = np.mean (features, axis=0)
        samples = np.zeros(num_samples)

        if method == 'l2':
            features -= mean_img

            for i in xrange(num_samples):
                # draw k samples
                indices = np.random.choice(np.arange(N), size=k, replace=True)
                samp = features[indices] 
                # find l2 distances to mean (0)
                samp_l2 = np.sum(samp*samp) / k
                samples[i] = samp_l2
		
	    # Run PCA to 128 features
	features += mean_img
	pca_model = PCA (n_components = 128)
	f_hat = pca_model.fit_transform (features)
	expl_var = sum (pca_model.explained_variance_ratio_ )	
	print "Explained variance of 4096 --> 128 features: ", expl_var

        with open('{0}_pca_128'.format(synset), 'w+') as f:
		pickle.dump(f_hat, f)
	
	return samples            


#-------------
def build_model(input_var,batch_size = None):
    net = {}
    net['input'] = InputLayer((batch_size, 3, 224, 224),input_var=input_var)
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

    # Finish at the 4096 level
    return net


def load_data():
    model = pickle.load(open('../weights/vgg19.pkl'))

    classes = model['synset words']
    mean_image= model['mean value']
    
    return model, classes, mean_image

def prepare_vgg16():

    input_var = T.tensor4('inputs') 
    net = build_model(input_var,batch_size=batch_size)
    # Load vgg(16) weights
    model, classes, mean_image = load_data()
    # update only up to the fully connected layer
    lasagne.layers.set_all_param_values(net['fc6'], model['param values'][:-4])
    return net, mean_image


batch_size = 128
if __name__ == '__main__':
        
    synsets = ['n02105056']
    if len(sys.argv) > 1 and (sys.argv[1] == '-p' or sys.argv[1] == '--pipe'): # pipe in synsets to use to stdin.
        synsets = sys.stdin.read().split('\n')[:-1]

    print synsets

    net, mean_image = prepare_vgg16()
    boot = Bootstrap(net)
    
    syn = synsets[0]        
    synsets.append(None)
    for i in xrange(1, len(synsets)):
	if os.path.exists(syn + "_samples"):
	    print "skipping {0}: already exists".format(syn)
	    syn = synsets[i]
	    continue

	print "starting new batch"        
	t0 = time.time()
       	_, features = boot.forward(syn, synsets[i])
       	tbefore = time.time()
	samples = boot.sample(syn, features)
	tafter = time.time() 
        with open("{0}_samples".format(syn), 'w+') as f:
            pickle.dump(samples, f)
        shutil.rmtree ('/mnt/data/{0}'.format(syn))
	t1 = time.time()
	print "Took {0:.3f} seconds to bootstrap".format(tafter - tbefore)
        print "took {0:.3f} seconds to run syns {1}".format(t1 - t0, syn)
        syn = synsets[i]
    
