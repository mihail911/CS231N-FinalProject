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
from gen_adver import FastGradient
from textwrap import wrap

import theano
import theano.tensor as T
from theano import pp
import time

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from lasagne.nonlinearities import softmax


def prep_image(url,mean_image):
    # ext = url.split('.')[-1]
    im = plt.imread(url, 'png')
    im = np.uint8(im*255)
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
    #im = im[::-1, :, :]
    im = im - mean_image[:,None,None]
    return rawim, floatX(im[np.newaxis])


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
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net



def load_data():
    model = pickle.load(open('../weights/vgg19.pkl'))

    classes = model['synset words']
    mean_image= model['mean value']
    
    return model, classes, mean_image


def download_val_images (num_ex, mean_image,gold_labels,start,end):
    ''' 
    Dynamically downloads sample images from ImageNet.  
    '''
    # index = urllib.urlopen('http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html').read()
    image_urls = index.split('<br>')
    allFiles = os.listdir('/mnt/val_images')
    final_labels = gold_labels.copy()
    np.random.seed(61)
    np.random.shuffle(image_urls)
    np.random.seed(61)
    np.random.shuffle(final_labels)
    result_labels = np.zeros(num_ex)
    valid_urls = []

    images = np.zeros ((num_ex, 3, 224, 224), dtype=np.float32)
    i = 0
    j = 0
    used=True
    rawim = np.zeros ((num_ex, 224, 224, 3), dtype=np.float32)
    tot = 0
    for j in xrange(start,end):
        # only use quick downloads on flickr
        # if 'static.flickr' not in im_url:
        #     continue
        im = allFiles[j]
        rawimTemp, result = prep_image (im, mean_image)
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
    
    
def load_gold_labels():
    f = open("../datasets/val_gold_labels.txt",'r')
    labels = []
    for line in f:
        labels.append(int(line))
        
    return np.asarray(labels)

# TODO: change num_images 32 to a more global backprop pass size
def find_adverserial_examples(tot_images=1,batch_size=1,start=0,end=1, log=True):
    cls = FastGradient(num_images = batch_size,input_dim=(3,224,224), eps=0.85, 
                   loss='softmax')
    if log:
        print "Finished creating Fast Gradient Sign Class......"

    model, classes, mean_image = load_data()
    if log:
        print "Finished loading data......"

    all_gold_labels = load_gold_labels()
    if log:
        print "Finished loading golden labels......"

    # rawim, images,gold_labels,valid_urls = download_val_images(tot_images, mean_image, gold_labels,start,end)
    # if log:
    #     print "Finished downloading images, normalizing them and extracting required number of images and labels......"

    input_var = T.tensor4('inputs')
    net = build_model(input_var,batch_size=batch_size)
    lasagne.layers.set_all_param_values(net['prob'], model['param values'])
    if log:
        print "Finished setting all the parameters......"

    if batch_size > tot_images:
        print "Input the correct batch size and/or the total images in the dataset"
        exit(1)

    rem = _num_images%batch_size
    num_iter = 0
    if rem == 0:
        num_iter = _num_images/batch_size
    else:
        num_iter = _num_images/batch_size + 1
    
    trueProb_dist = []
    advProb_dist = []
    actualAdvProb = []
    actualAdvLabel = []
    actualTrueProb = []
    actualTrueLabel = []
    advHighUrl = []
    advLowUrl = []

    sameAdv = []
    sameTrue = []
    advCount = 0
    nonAdvCount = 0

    cur_start = start
    finalEnd = end
    cur_end = batch_size
    for i in xrange(num_iter):
        if log:
            print "Started Batch Iteration " + str(i+1) + " out of " + str(num_iter)
        print "Current Start image is " + str(cur_start)
        print "Current End image is " + str(cur_end)
        rawim, images,gold_labels,valid_urls = download_val_images(cur_end - cur_start, mean_image,
                                                                   all_gold_labels,cur_start,cur_end)
        if log:
            print "Finished downloading images, normalizing them and extracting required number of images and labels for iteration " + str(i+1) +  " ......"
        t0 = time.time()
        curSet = np.copy(images[i*batch_size:(i+1)*batch_size,:,:,:])
        true_prob = np.array(lasagne.layers.get_output(net['prob'], curSet, deterministic=True).eval(), dtype=np.float32)
        true_top5 = np.argsort(true_prob,axis=1)[:,-1:-6:-1]
        trueProb_dist += list(true_prob[np.arange(batch_size),true_top5[:,0]])
        t1 = time.time()
        print "Time taken for forward pass of {0} : {1} seconds".format(batch_size, t1 - t0)

        if log:
            print "Finished forward pass to get the True Image class probabilites......"

        newim = rawim[i*batch_size:(i+1)*batch_size,:,:,:].transpose(0,3,1,2)
        
        back_batch_sz = batch_size
        ind = 0
        final = np.zeros_like(newim)

        while ind < batch_size:
            bt1 = time.time()
            print "Back pass for batch {0}-{1}".format(ind, ind + back_batch_sz)
            small_im = np.copy(newim[ind : ind + back_batch_sz, :, :, :])	

            start_idx = i*batch_size + ind
            end_idx = min(start_idx + back_batch_sz, (i+1)*batch_size)

            final[ind : ind + back_batch_sz, :, :, :] = cls.adExample(small_im, 
                np.array(gold_labels[start_idx:end_idx]),
                model['param values'],
                net['prob'],
                input_var)
            bt2 = time.time()
            print "Took {0} seconds for mini-mini-batch back".format(bt2 - bt1)
            ind += back_batch_sz

        if log:
            print "Finished generation of adverserial examples for current batch......"

        final = final - mean_image[None,:,None,None]
        
        final = final.astype(np.float32)
        print final.dtype

        adv_prob = np.array(lasagne.layers.get_output(net['prob'], final, deterministic=True).eval(), dtype=np.float32)
        adv_top5 = np.argsort(adv_prob,axis=1)[:,-1:-6:-1]
        advProb_dist += list(adv_prob[np.arange(batch_size),adv_top5[:,0]])

        final = final + mean_image[None,:,None,None]
       
        if log:
            print "Finished forward pass for adverserial examples"

        curCount = 0
        for k in xrange(batch_size):
            advLabel = adv_top5[k,0]
            trueLabel = true_top5[k,0]
            if advLabel != trueLabel:
                actualAdvProb.append(adv_prob[k,advLabel])
                actualAdvLabel.append(advLabel)
                actualTrueProb.append(true_prob[k,trueLabel])
                actualTrueLabel.append(trueLabel)
                imName = ''
                if adv_prob[k,advLabel] >= 0.5:
                    advHighUrl.append(valid_urls[i*batch_size+k])
                    imName = '/mnt/advResults/high_{0}_{1:.2f}_{2}_{3:.2f}.png'.format(str(classes[trueLabel]),true_prob[k,trueLabel],
                                                                           str(classes[advLabel]),adv_prob[k,advLabel])
                else:
                    advLowUrl.append(valid_urls[i*batch_size+k])
                    imName = '/mnt/advResults/{0}_{1:.2f}_{2}_{3:.2f}.png'.format(str(classes[trueLabel]),true_prob[k,trueLabel],
                                                                           str(classes[advLabel]),adv_prob[k,advLabel])
                scipy.misc.imsave(imName, final[curCount,:,:,:].transpose(1,2,0).astype('uint8'))
                advCount += 1
            else:
                sameAdv.append(adv_prob[k,advLabel])
                sameTrue.append(true_prob[k,trueLabel])
                imName = '/mnt/advResults/nonAdvImage_{0}_{1:.2f}_{2}_{3:.2f}.png'.format(str(classes[trueLabel]),true_prob[k,trueLabel],
                                                                           str(classes[advLabel]),adv_prob[k,advLabel])
                scipy.misc.imsave(imName, final[curCount,:,:,:].transpose(1,2,0).astype('uint8'))
                nonAdvCount += 1
                
            curCount += 1

        cur_start += batch_size
        cur_end += batch_size
        if cur_end > finalEnd:
            curEnd = finalEnd
        if log:
            print "Finished Batch Iteration " + str(i+1) + " out of " + str(num_iter)

    #   Convert the lists to arrays
    binSplit = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    actualAdvProb = np.asarray(actualAdvProb)
    actualAdvLabel = np.asarray(actualAdvLabel)
    actualTrueProb = np.asarray(actualTrueProb)
    actualTrueLabel = np.asarray(actualTrueLabel)
    advHighUrl = np.asarray(advHighUrl)
    advLowUrl = np.asarray(advLowUrl)

    # Open files to write the data collected into
    text2 = open("/mnt/advResults/advLowUrl.txt","w")
    text1 = open("/mnt/advResults/advHighUrl.txt","w")
    text3 = open("/mnt/advResults/trueProb_dist.txt","w")
    text4 = open("/mnt/advResults/advProb_dist.txt","w")

    #   Index into the arrays to find the elements having high confidence adervarial examples
    highAdvProb = actualAdvProb[actualAdvProb >= 0.5]
    highAdvLabel = actualAdvLabel[actualAdvProb >= 0.5]
    highTrueProb = actualTrueProb[actualAdvProb >= 0.5]
    highTrueLabel = actualTrueLabel[actualAdvProb >= 0.5]

    #   Save the data collected to files
    np.savetxt(text1,advHighUrl, fmt='%s')
    np.savetxt(text2,advLowUrl, fmt='%s')


    highTrueProb = list(highTrueProb)

    #   Plot histograms for variour purposes
    plt.hist(trueProb_dist,bins=binSplit)
    plt.title('Frequency Distribution of Confidences of VGG Predictions')
    plt.xlabel('Confidences')
    plt.ylabel('Count')
    plt.savefig("/mnt/advResults/true_prob_dist.pdf")
    plt.figure()

    trueProb_dist = np.asarray(trueProb_dist)

    plt.hist(advProb_dist,bins=binSplit)
    title = "Frequency Distribution of Confidences of VGG Adverserial Example Predictions"
    plt.title('\n'.join(wrap(title,60)))
    plt.xlabel('Confidences')
    plt.ylabel('Count')
    plt.savefig("/mnt/advResults/adv_prob_dist.pdf")
    plt.figure()

    advProb_dist = np.asarray(advProb_dist)

    plt.hist(list(actualTrueProb),bins=binSplit)
    title = "Frequency Distribution of true class predictions having some type of adverserial image"
    plt.title('\n'.join(wrap(title,60)))
    plt.xlabel('Confidences of true predictions')
    plt.ylabel('Counts')
    plt.savefig("/mnt/advResults/true_having_some_adv.pdf")
    plt.figure()

    plt.hist(highTrueProb,bins=binSplit)
    title = "Frequency Distribution of true class predictions having an adverserial image with >50% confidence"
    plt.title('\n'.join(wrap(title,60)))
    plt.xlabel('Confidences of true predictions')
    plt.ylabel('Counts')
    plt.savefig("/mnt/advResults/true_having_high_adv.pdf")
    plt.figure()

    plt.hist(sameTrue,bins=binSplit)
    title = "Frequency Distribution of true class predictions not perturbed in their adverserial examples"
    plt.title('\n'.join(wrap(title,60)))
    plt.xlabel('Confidences of true predictions')
    plt.ylabel('Counts')
    plt.savefig("/mnt/advResults/true_having_same_adv.pdf")

    np.savetxt(text3,trueProb_dist, fmt='%1.2f')
    np.savetxt(text4,advProb_dist, fmt='%1.2f')
    
def run_fsg_adverserial(tot_images=1,batch_size=1,start=0,end=1):
    find_adverserial_examples(tot_images=tot_images,batch_size=batch_size,start=start,end=end,log=True)
   
def main():
    t1 = time.time()
    run_fsg_adverserial(tot_images=64*_num_splits,batch_size=32, start=0,end=64*_num_splits)
    t2 = time.time()
    print "Time taken to generate {0} adv examples is {1} seconds".format(64*_num_splits, t2-t1)

_num_images = 50000
_num_splits = 782
if __name__ == "__main__":
	main()
