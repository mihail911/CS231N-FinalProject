
# coding: utf-8

# Does density estimation

import numpy as np
from scipy import stats
import matplotlib as plt
import cPickle as pickle
from sklearn.decomposition import PCA
import sys

import autoencoder
import multiprocessing as mp


import theano
import theano.tensor as T
from theano import pp

import lasagne
from time import time

def plot_density (kernel):
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = kernel(positions).T
    Z = np.reshape(values, X.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
              extent=[xmin, xmax, ymin, ymax])
    ax.plot(x_vals, y_vals, 'k.', markersize=2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.title('Kernel Density Estimation')
    plt.savefig('kde.png', dpi=500)
    #plt.show()
    

# Now let's try to load PCA features and run the autoencoder on them.
# 


_prefix = '../../../231n_results/'

def load_pca_feat (syn='n02119789'):
    with open(_prefix + 'train_features/{0}_pca_128'.format(syn), 'rb') as f:
        pca_feat = pickle.load(f)
        
    return pca_feat

def load_pca_model (syn = 'n02119789'):
    
    
    with open(_prefix + 'train_pca/{0}_pca_model'.format(syn), 'rb') as f:
        pca_model = pickle.load(f)
        
    return pca_model

def parallelize_batch (fn, args, batch_size):
    """ 
    Parallelizes the function into the given batches.
    """
    pool = mp.Pool(8)
    
    for i in np.arange(0, len(args), batch_size):        
        yield pool.map(fn, args[i:i+batch_size])
        
    pool.close()
    pool.join()


def prep_encoder():
    net, input_var, target_var, init_weights = autoencoder.buildEncoder()
    half_network = lasagne.layers.get_output (net['h3'], input_var, deterministic=True)
    predict_fn = theano.function([input_var], half_network)
    
    return net, predict_fn

# Load the autoencoder weights from file 
def load_weights_for(syn):
    
    ''' Loads weights for a synset into the autoencoder '''
    
    with open(_prefix + 'train_autoencode/{0}.pkl'.format('n02119789'), 'rb') as f:
        weights = pickle.load(f)
    return weights
    
# Get prediction

def load_weights_and_pca_feat (syn):
    weights = load_weights_for (syn)
    feat = load_pca_feat (syn)
    return feat, weights

def form_density_priors (synsets, vgg_out = None):
    '''
    Forms the priors as described in the paper.
    Input:
    - vgg_out: the forward pass of vgg up to 4096.  
    '''
    
    O, C = 128, len(synsets)
    
    encoder,predict_fn = prep_encoder()
    if vgg_out is None:
        # random input
        N, I = 1500, 4096
        vgg_out = np.random.random((N, I))
    else:
        N, I = vgg_out.shape

    args = synsets

    # Get output of lasagne function forward pass ...

    projected = np.zeros ((C, N, O))

    t0 = time()
    i = 0
    for features in parallelize_batch (load_pca_model, args, 8):
        for m in features:
            projected[i, :, :] = m.transform(vgg_out)
            i += 1
        print i, '---'

    t1 = time()

    print t1 - t0, " seconds elapsed for pca compression"

    # enc_out = np.zeros ((C, N, 8))
    # Run the autoencoder forward pass, making sure to calculate the pca features too...
    j = 0
    density = np.zeros ((C, N))

    for entry in parallelize_batch (load_weights_and_pca_feat, args, 8):

        for feat, w in entry:
            lasagne.layers.set_all_param_values (encoder['h0_inv'], w)
            class_out = predict_fn (feat)
            print "shape: ", feat.shape
            enc_out =  predict_fn (projected[j, :, :])
            print "enc_out: ", enc_out
            # form density estimate
            kernel = stats.gaussian_kde(class_out.T, 'silverman')
            # scale density by number of support examples
            density[j, :] =  kernel(enc_out.T) * (1.0 * feat.shape[0])

            j += 1
        print '---'

    print density.shape

    t2 = time()

    print "Total time elapsed: {0:.3f}".format(t2 - t0)
    print density
    return density



def run(synsets):
    form_density_priors(synsets)

if __name__ == '__main__':
    
    synsets = ['n02105056']
    if len(sys.argv) > 1 and (sys.argv[1] == '-p' or sys.argv[1] == '--pipe'): # pipe in synsets to use to stdin.
        synsets = sys.stdin.read().split('\n')[:-1]

    print synsets
    run(synsets)



