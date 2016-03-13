# coding: utf-8
import os
import sys
sys.path.insert(0, os.path.abspath("~/.local/lib/python2.7/site-packages"))
sys.path.append(os.path.abspath('..'))

import lasagne
from util.data_utils import get_CIFAR10_data

os.environ['THEANO_FLAGS'] = 'floatX=float32'


import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano import pp
import time

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, InverseLayer

import multiprocessing as mp

import cPickle as pickle
from lasagne.utils import floatX
from lasagne.updates import adam
from lasagne.nonlinearities import softmax

import glob 

_prefix = "../../../231n_results/"

def load_one (args):
    ''' Parallel task to load pca features from disk '''
    slice_no, syn_link = args

    try:
        with open(syn_link, 'r') as f:
            X = floatX(pickle.load(f))
    except IOError as e:
        print "Error :" + str(e)
        return None, slice_no

    return X, slice_no

def load_data (fake=False, synsets = None):
    # TODO: import diskreader, etc...

    if fake:
        X = floatX(np.random.random((1024, 128)))
    else:

        if synsets:
            syns_path = [_prefix + "train_features/{0}_pca_128".format(syn) for syn in synsets]
        else:    
            syns_path = glob.glob(_prefix + 'train_features/*pca_128')
        
        print syns_path

        N = len(syns_path)
        
        before = time.time()
        print "starting in parallel..."
        pool = mp.Pool(8)
        args = [(i, syns_path[i]) for i in xrange(N)]
        print len(args)


        vectors = pool.map(load_one, args)

        counter = 0
        print len(vectors)
        for x, ind in vectors:
            if x is None:
                print "Don't have ind {0}: {1}".format(ind, syns_path[ind])
                counter += 1
        after = time.time()
        print "Loaded {0} pca features in {1:.3f} seconds".format(N - counter, after - before)

    y = X
    return X, y


def load_cifar10():
    img_data = get_CIFAR10_data()
    X_val = img_data['X_val']
    X_val = np.array( X_val.reshape((1000, -1)), dtype=np.float32)
    print X_val.shape
    data = X_val
    y = data
    return data, y 

def buildEncoder(hidden_sizes=[64, 32, 16, 8], input_sz=128):

    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')

    names = ['h{0}'.format(i) for i in xrange(len(hidden_sizes))]
    inv_names = ['h{0}_inv'.format(i) for i in xrange(len(hidden_sizes))][::-1]
    net = {}

    print names, inv_names

    net['input'] = InputLayer ((None, input_sz), input_var=input_var)

    prev = net['input']
    for i, name in enumerate(names):
        net[name] = DenseLayer(prev, num_units=hidden_sizes[i], 
                                     nonlinearity=T.nnet.relu,
                                     W=lasagne.init.HeNormal(gain='relu') )
        prev = net[name]


    # Build reverse layers  
    for j, name in enumerate(inv_names):
        print names[-j-1]
        net[name] = InverseLayer(prev, net[names[-j-1]])
        prev = net[name]

    return net, input_var, target_var

def buildFunctions(net, input_var, target_var):

    params = lasagne.layers.get_all_params(net['h0_inv'], trainable=True)
    out = lasagne.layers.get_output(net['h0_inv'], deterministic=True)

    loss = lasagne.objectives.squared_error(out, target_var)
    adam_update = adam (loss.mean(), params)

    train_function = theano.function([input_var, target_var], loss, updates=adam_update)

    return train_function


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Get minibatch for data.
    :param inputs:
    :param targets:
    :param batchsize:
    :param shuffle:
    :return:
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def train(x, y, train_function, num_epochs=30):
    batch_size = 512
    print type(x)

    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0.0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(x, y, batch_size, shuffle=True):
            inputs, targets = batch
            train_function(inputs, targets)
            train_batches += 1
        print 'epoch {0} done in time {1}'.format(epoch, time.time() - start_time)
    

def getEncodedOutput (net, data):
    compressed = np.array(lasagne.layers.get_output(
            net['h1_inv'], data, deterministic=True).eval())

    compressed = compressed.reshape((1000, 3, 32, 32))
    
    return compressed

def imshow_noax(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')


def visualize (data, compressed):
    normed_data = data.reshape((1000, 3, 32, 32))
    normed_data = np.transpose(normed_data, (0, 2, 3, 1))

    imshow_noax(np.transpose(compressed[0], (1, 2, 0)))
    imshow_noax(normed_data[0])

    
def run(synsets):

    net, input_var, target_var = buildEncoder()

    train_fn = buildFunctions(net, input_var, target_var)


    for _ in xrange(1):
        print "loading iter {0}".format(_)
        X, y = load_data(True)        
        train(X, y, train_fn)
        print "Done with iteration {0}".format(_)

if __name__ == '__main__':

    synsets = ['n02105056']
    if len(sys.argv) > 1 and (sys.argv[1] == '-p' or sys.argv[1] == '--pipe'): # pipe in synsets to use to stdin.
        synsets = sys.stdin.read().split('\n')[:-1]

    print synsets
    run(synsets)




