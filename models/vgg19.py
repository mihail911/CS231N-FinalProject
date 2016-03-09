# coding: utf-8

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import urllib
import io
import skimage.transform
import sys
import theano
import theano.tensor as T
import time
import os

# for mihail
# sys.path.append("/Users/mihaileric/Documents/Research/Lasagne")
# allow imports from directory above
sys.path.append("..")


import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.conv import Conv2DLayer as ConvLayer
from lasagne.layers.pool import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from lasagne.nonlinearities import softmax
from util.util import load_dataset_batch, compute_accuracy_batch, get_image_id_mapping

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
        #im = im[::-1, :, :]
        im = im - mean_image[:,None,None]
        return rawim, floatX(im[np.newaxis])

    except:
        # Abort
        print "skipping url " + url
        return None, np.zeros((1,))


def train_and_predict_funcs(weights=None, update="nesterov", regularization=0.0):
    """
    Create theano functions for computing loss, accuracy, etc. for given model
    :param model:
    :param update: Update parameter to use for training. Select from among
                    "nesterov", "sgd", "rmsprop", etc.
    :return:
    """
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    model = build_model(input_var)["prob"]

    #print "Weights before loading..."
    #print lasagne.layers.get_all_param_values(model, trainable=True)
    with open("weights_before.txt", "a") as f:
        f.write("Weights before loading...\n")
        f.write(str(lasagne.layers.get_all_param_values(model, trainable=True)))

    # Load pre-trained weights
    if weights != None:
        lasagne.layers.set_all_param_values(model, weights)

    with open("weights_after.txt", "a") as f:
        f.write("Weights after loading...\n")
        f.write(str(lasagne.layers.get_all_param_values(model, trainable=True)))

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(model)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # TODO: Add regularization to the loss

    params = lasagne.layers.get_all_params(model, trainable=True)
    if update == "nesterov":
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)
    else:
        pass

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(model, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # theano function giving output label for given input
    predict_fn = theano.function([input_var], test_prediction)

    return train_fn, val_fn, predict_fn, model


# Model to be trained/tested against
def build_model(input_var):
    """ 
    Builds the classic VGG-19 model using Lasagne wrapper to Theano.
    """
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224), input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1)
    input = np.random.randn(2, 3, 224, 224).astype(np.float32)
    # out1 = lasagne.layers.get_output(net['conv1_1'], input)
    # print "Output: ", out1.eval()
    # print "Output: ", out1.shape.eval()
#    fout1 = theano.function([input_var], out1)
#    print
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

def load_weights(path):
    with open(path, "r") as f:
        model = pickle.load(f)

    classes = model["synset words"]
    mean_image = model["mean value"]
    values = model["param values"]

    idx_to_class_map = {}
    class_to_idx_map = {}
    for idx, c in enumerate(classes):
        idx_to_class_map[idx] = c
        class_to_idx_map[c] = idx
    
    return idx_to_class_map, class_to_idx_map, classes, mean_image, values

# Validation Set
def download_val_images (num_ex, mean_image):
    ''' 
    Dynamically downloads sample images from ImageNet.  
    '''
    index = urllib.urlopen('http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html').read()
    image_urls = index.split('<br>')

    np.random.seed(19)
    np.random.shuffle(image_urls)

    images = np.zeros((num_ex, 3, 224, 224), dtype=np.float32)
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

# Process test images and print top 5 labels
def run_forward(images):

    print images.shape
    prob = np.array(lasagne.layers.get_output(net['prob'], images, deterministic=True).eval())


def logStats(filename, time, acc, batch_acc, num_ex):
    with open(filename, 'a') as f:
        f.write("Time elapsed {0}, Current running accuracy {1}, Batch accuracy {2}, Num Ex {3}\n".format(str(time), str(acc), str(batch_acc), str(num_ex)))


def compute_accuracy(data_dir, val_filename, lower_idx, upper_idx):
    # for normalizing images and loading pre-trained weights
    idx_to_class_map, _, _, mean_image, param_values = load_weights("../datasets/vgg19.pkl")

    # TODO: Check that params are being set
    _, val_fn, predict_fn, _ = train_and_predict_funcs(weights=param_values)
    
    #img_id_mapping = get_image_id_mapping("/afs/ir/users/m/e/meric/Documents/CS231N/CS231N-FinalProject/datasets/parsedData.txt")

    batch_size = 10
    # TODO: Change hard-coding of number of examples in data
    num_ex = float(upper_idx - lower_idx) #50000.0
    batch_frac = batch_size / num_ex
    total_acc = 0.
    total_ex = 0
    for data_batch, labels_batch in load_dataset_batch(data_dir,
                                            val_filename, batch_size, mean_image, lower_idx, upper_idx):
        start_time = time.time()
        print "Computed accuracy on {0}".format(str(total_ex))
        batch_acc = compute_accuracy_batch(val_fn, predict_fn, data_batch, labels_batch, idx_to_class_map)
        total_acc += batch_frac*batch_acc
        time_elapsed = time.time() - start_time
        print "Time to compute batch acc: ", str(time_elapsed)
        print "Batch accuracy: ", str(batch_acc)
        print "Total accuracy so far: ", str(total_acc)
        logStats("../logs/VanillaVGG19Run_"+str(lower_idx)+"_"+str(upper_idx)+".log", time_elapsed, total_acc, batch_acc, total_ex)

        total_ex += batch_size

    print "Accuracy for run: {0}".format(str(total_acc))

    return total_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="get range of image values")
    parser.add_argument("--lower", type=int, help="lower bound on img idx")
    parser.add_argument("--upper", type=int, help="upper bound on img idx") 
    args = parser.parse_args() 
    # TODO: fill with your own
    data_dir = "/farmshare/user_data/meric"
    val_filename = "/afs/ir/users/m/e/meric/Documents/CS231N/CS231N-FinalProject/datasets/val_gold_labels.txt"
    compute_accuracy(data_dir, val_filename, args.lower, args.upper)
