import numpy as np
import matplotlib.pyplot as plt
import pickle
import urllib
import io
import skimage.transform
from textwrap import wrap
import os

import theano
import theano.tensor as T
from theano import pp
import time

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, get_output
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from lasagne.nonlinearities import softmax


def build_model(batch_size = None,input_var = None):
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
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net


def make_fooling_image(X, target_y, model,input_var,var_y):

    print "Starting to make fooling images"
    X_fooling = X.copy()
    lr = 1000
    for i in xrange(100):
        dX = model_train(X_fooling, target_y,input_var, var_y, model)
        X_fooling += lr*dX
        print "Completed iteration {0}".format(i+1)
    
    
    
    return X_fooling

def model_train(X,target_y, input_var, var_y, model):
    dX = train_function(X,target_y)
    return dX
    
    
    
def load_labels():
    f = open("../datasets/val_gold_labels.txt",'r')
    labels = []
    for line in f:
        labels.append(int(line))

    return np.asarray(labels)


def prep_image(url,mean_image):
    # ext = url.split('.')[-1]
    im = plt.imread(url,'JPEG')
    # Resize so smallest dim = 256, preserving aspect ratio
#     print url
    if im.ndim < 3:
        return None, None
    
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



def download_val_images (num_ex, mean_image,gold_labels,start,end):
    ''' 
    Dynamically downloads sample images from ImageNet.  
    '''
    # index = urllib.urlopen('http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html').read()
#     image_urls = index.split('<br>')
    imPath = '/srv/zfs01/user_data/nipuna1/val_data'
    allFiles = os.listdir(imPath)
    final_labels = gold_labels.copy()
    result_labels = np.zeros(num_ex)
    valid_urls = []

    images = np.zeros ((num_ex, 3, 224, 224), dtype=np.float32)
    i = 0
    j = 0
    used=True
    rawim = np.zeros ((num_ex, 224, 224, 3), dtype=np.float32)
    tot = 0
    for j in xrange(start,end):
        im = allFiles[j]
        print im
        rawimTemp, result = prep_image (imPath + '/' + im, mean_image)
        if result is None:
            continue
    
        if result.any():
            images[i,:,:,:] = result
            result_labels[i] = final_labels[j]
            rawim[i,:,:,:] = rawimTemp
            i += 1
            tot += 1
            valid_urls.append(im)
           
        if i >= num_ex: 
            break
        if tot >= (end-start):
            break
        j += 1
            
    return rawim,images,result_labels,valid_urls


def load_data():
    model = pickle.load(open('../weights/vgg19.pkl'))

    classes = model['synset words']
    mean_image= model['mean value']
    
    return model, classes, mean_image



batch_size = 2
num_ex = 2


model, classes, mean_image = load_data()
print "Finished loading data......"

gold_labels = load_labels()
print "Finished loading golden labels......"

input_var = T.tensor4('inputs')
var_y = T.ivector('y')
net = build_model(batch_size=batch_size, input_var=input_var)
lasagne.layers.set_all_param_values(net['prob'], model['param values'])
print "Finished creating the network and setting all the parameters......"

prediction = lasagne.layers.get_output(net['prob'])
loss = lasagne.objectives.categorical_crossentropy(prediction, var_y)
loss = loss.mean()
dX = theano.grad(loss, input_var)
train_function = theano.function([input_var, var_y],dX,allow_input_downcast=True)
print "Created the training function"

start = 0
end = 2
rawim,images,result_labels,valid_urls = download_val_images (num_ex, mean_image,gold_labels,start,end)
print "Finished downloading images, normalizing them and extracting required number of images and labels......"

np.random.seed(61)
target_y = np.random.choice(result_labels,num_ex)
print "Original Classes are " + str(result_labels[-1:-batch_size]) 
print "Converting to following classes: " + str(target_y)

foolingX = make_fooling_image(images, target_y, net,input_var,var_y)

print "Completed making Fooling Images"

for i in xrange(batch_size):
    true_prob = np.array(lasagne.layers.get_output(net['prob'], curSet, deterministic=True).eval())
    print 'Making forward pass on true image'
    true_top5 = np.argsort(true_prob,axis=1)[:,-1:-6:-1]
    adv_prob = np.array(lasagne.layers.get_output(net['prob'], foolingX, deterministic=True).eval())
    print "Making forward pass on fooling image"
    adv_top5 = np.argsort(adv_prob,axis=1)[:,-1:-6:-1]
    finalImage = foolingX[i,:,:,:] + mean_image
    
    advLabel = adv_top5[k,0]
    trueLabel = true_top5[k,0]
    if advLabel != trueLabel:
        origName = '/mnt/advResults/orig_{0}_{1:.2f}_{2}_{3:.2f}.png'format(str(classes[trueLabel]),true_prob[k,trueLabel],
                                                                       str(classes[advLabel]),adv_prob[k,advLabel])
        imName = '/mnt/advResults/{0}_{1:.2f}_{2}_{3:.2f}.png'.format(str(classes[trueLabel]),true_prob[k,trueLabel],
                                                                       str(classes[advLabel]),adv_prob[k,advLabel])
        scipy.misc.imsave(origName, rawim[k,:,:,:].transpose(1,2,0).astype('uint8'))
        scipy.misc.imsave(imName, final[k,:,:,:].transpose(1,2,0).astype('uint8'))
            