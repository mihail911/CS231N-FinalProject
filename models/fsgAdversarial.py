import numpy as np
import matplotlib.pyplot as plt
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
    ext = url.split('.')[-1]
    im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)
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
    index = urllib.urlopen('http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html').read()
    image_urls = index.split('<br>')
    final_labels = gold_labels.copy()
    np.random.seed(61)
    np.random.shuffle(image_urls)
    np.random.seed(61)
    np.random.shuffle(final_labels)
    result_labels = np.zeros(num_ex)
    valid_urls = []
    
    allImages = []
    path_to_im = " /farmshare/user_data/meric"

    images = np.zeros ((num_ex, 3, 224, 224), dtype=np.float32)
    i = 0
    j = 0
    used=True
    rawim = np.zeros ((num_ex, 224, 224, 3), dtype=np.float32)
    tot = 0
    for im_url in image_urls[start:end+500]:
        # only use quick downloads on flickr
        if 'static.flickr' not in im_url:
            continue
            
        rawimTemp, result = prep_image (im_url, mean_image)
        if result is None:
            continue
    
        if result.any():
            images[i,:,:,:] = result
            result_labels[i] = final_labels[j]
            rawim[i,:,:,:] = rawimTemp
            i += 1
            tot += 1
            valid_urls.append(im_url)
           
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

def find_adverserial_examples(tot_images=1,batch_size=1,start=0,end=1, log=True):
    cls = FastGradient(num_images = batch_size,input_dim=(3,224,224), eps=0.85, 
                   loss='softmax')
    if log:
        print "Finished creating Fast Gradient Sign Class......"

    model, classes, mean_image = load_data()
    if log:
        print "Finished loading data......"

    gold_labels = load_gold_labels()
    if log:
        print "Finished loading golden labels......"

    rawim, images,gold_labels,valid_urls = download_val_images(tot_images, mean_image, gold_labels,start,end)
    if log:
        print "Finished downloading images, normalizing them and extracting required number of images and labels......"

    print len(valid_urls)
    input_var = T.tensor4('inputs')
    net = build_model(input_var,batch_size=batch_size)
    lasagne.layers.set_all_param_values(net['prob'], model['param values'])
    if log:
        print "Finished setting all the parameters......"

    if batch_size > tot_images:
        print "Input the correct batch size and/or the total images in the dataset"
        exit(1)

    num_iter = (len(valid_urls))/batch_size
    trueProb_dist = []
    advProb_dist = []
    actualAdvProb = []
    actualAdvLabel = []
    actualTrueProb = []
    actualTrueLabel = []
    advUrl = []

    sameAdv = []
    sameTrue = []

    for i in xrange(num_iter):
        if log:
            print "Started Batch Iteration " + str(i+1) + " out of " + str(num_iter)

        curSet = images[i*batch_size:(i+1)*batch_size,:,:,:]
        true_prob = np.array(lasagne.layers.get_output(net['prob'], curSet, deterministic=True).eval())
        true_top5 = np.argsort(true_prob,axis=1)[:,-1:-6:-1]
        trueProb_dist += list(true_prob[np.arange(batch_size),true_top5[:,0]])

        if log:
            print "Finished forward pass to get the True Image class probabilites......"

        newim = rawim[i*batch_size:(i+1)*batch_size,:,:,:].transpose(0,3,1,2)
        final = cls.adExample(newim,np.array(gold_labels[i*batch_size:(i+1)*batch_size]),model['param values'],net['prob'],input_var)
        if log:
            print "Finished generation of adverserial examples for current batch......"

        final = final - mean_image[None,:,None,None]
        adv_prob = np.array(lasagne.layers.get_output(net['prob'], final, deterministic=True).eval())
        adv_top5 = np.argsort(adv_prob,axis=1)[:,-1:-6:-1]
        advProb_dist += list(adv_prob[np.arange(batch_size),adv_top5[:,0]])

        final = final + mean_image[None,:,None,None]
        if log:
            print "Finished forward pass for adverserial examples"

        for k in xrange(batch_size):
            advLabel = adv_top5[k,0]
            trueLabel = true_top5[k,0]
            if advLabel != trueLabel:
                actualAdvProb.append(adv_prob[k,advLabel])
                actualAdvLabel.append(advLabel)
                actualTrueProb.append(true_prob[k,trueLabel])
                actualTrueLabel.append(trueLabel)
                advUrl.append(valid_urls[i*batch_size+k])
            else:
                sameAdv.append(adv_prob[k,advLabel])
                sameTrue.append(true_prob[k,trueLabel])


#            plt.subplot(121)
#            plt.imshow(rawim[i*batch_size+k,:,:,:].astype('uint8'))
#            for n, label in enumerate(true_top5[k,:]):
#                plt.text(0,260,'Original Image')
#                plt.text(0, 280 + n * 20, '{}. {} {} %'.format(n+1, classes[label],true_prob[k,label]*100), fontsize=12)
#
#            plt.subplot(122)
#            plt.imshow(final[k,:,:,:].transpose(1,2,0).astype('uint8'))
#            for n, label in enumerate(adv_top5[k,:]):
#                plt.text(0,260,'Adverserial Image')
#                plt.text(340, 280 + n * 20, '{}. {} {} %'.format(n+1, classes[label],adv_prob[k,label]*100), fontsize=12)


 #           plt.show() 

        if log:
            print "Finished Batch Iteration " + str(i+1) + " out of " + str(num_iter)

    #   Convert the lists to arrays
    binSplit = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    actualAdvProb = np.asarray(actualAdvProb)
    actualAdvLabel = np.asarray(actualAdvLabel)
    actualTrueProb = np.asarray(actualTrueProb)
    actualTrueLabel = np.asarray(actualTrueLabel)
    advUrl = np.asarray(advUrl)

    # Open files to write the data collected into
    text1 = open("highAdvProb1.txt","w")
    text2 = open("highAdvLabel1.txt","w")
    text3 = open("highTrueProb1.txt","w")
    text4 = open("highTrueLabel1.txt","w")
    text5 = open("highAdvUrl1.txt","w")
    text6 = open("trueProb_dist1.txt","w")
    text7 = open("advProb_dist1.txt","w")

    #   Index into the arrays to find the elements having high confidence adervarial examples
    highAdvProb = actualAdvProb[actualAdvProb >= 0.5]
    highAdvLabel = actualAdvLabel[actualAdvProb >= 0.5]
    highTrueProb = actualTrueProb[actualAdvProb >= 0.5]
    highTrueLabel = actualTrueLabel[actualAdvProb >= 0.5]
    advUrl = advUrl[actualAdvProb >= 0.5]

    #   Save the data collected to files
    np.savetxt(text1,highAdvProb, fmt='%1.8f')
    np.savetxt(text2,highAdvLabel, fmt='%s')
    np.savetxt(text3,highTrueProb, fmt='%1.8f')
    np.savetxt(text4,highTrueLabel, fmt='%s')
    np.savetxt(text5,advUrl, fmt='%s')


    highTrueProb = list(highTrueProb)

    #   Plot histograms for variour purposes
    plt.hist(trueProb_dist,bins=binSplit)
    plt.title('Frequency Distribution of Confidences of VGG Predictions')
    plt.xlabel('Confidences')
    plt.ylabel('Count')
    plt.savefig("true_prob_dist-1.pdf")
    plt.figure()

    trueProb_dist = np.asarray(trueProb_dist)

    plt.hist(advProb_dist,bins=binSplit)
    title = "Frequency Distribution of Confidences of VGG Adverserial Example Predictions"
    plt.title('\n'.join(wrap(title,60)))
    plt.xlabel('Confidences')
    plt.ylabel('Count')
    plt.savefig("adv_prob_dist-1.pdf")
    plt.figure()

    advProb_dist = np.asarray(advProb_dist)

    plt.hist(list(actualTrueProb),bins=binSplit)
    title = "Frequency Distribution of true class predictions having some type of adverserial image"
    plt.title('\n'.join(wrap(title,60)))
    plt.xlabel('Confidences of true predictions')
    plt.ylabel('Counts')
    plt.savefig("true_having_some_adv-1.pdf")
    plt.figure()

    plt.hist(highTrueProb,bins=binSplit)
    title = "Frequency Distribution of true class predictions having an adverserial image with >50% confidence"
    plt.title('\n'.join(wrap(title,60)))
    plt.xlabel('Confidences of true predictions')
    plt.ylabel('Counts')
    plt.savefig("true_having_high_adv-1.pdf")
    plt.figure()

    plt.hist(sameTrue,bins=binSplit)
    title = "Frequency Distribution of true class predictions not perturbed in their adverserial examples"
    plt.title('\n'.join(wrap(title,60)))
    plt.xlabel('Confidences of true predictions')
    plt.ylabel('Counts')
    plt.savefig("true_having_same_adv-1.pdf")

    np.savetxt(text6,trueProb_dist, fmt='%1.8f')
    np.savetxt(text7,advProb_dist, fmt='%1.8f')
    
def run_fsg_adverserial(tot_images=1,batch_size=1,start=0,end=1):
    find_adverserial_examples(tot_images=tot_images,batch_size=batch_size,start=start,end=end,log=True)
   
def main():
    t1 = time.time()
    run_fsg_adverserial(tot_images=128,batch_size=64, start=0,end=128)
    t2 = time.time()
    print "Time taken to execute program is " + str(t2-t1) + " seconds"


if __name__ == "__main__":
    main()
