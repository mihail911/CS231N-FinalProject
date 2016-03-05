"""
A series of util functions for different aspects of project.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import operator
import skimage.transform
import sys
# Janky work around for making sure the module can find 'Lasagne'
sys.path.append("/Users/mihaileric/Documents/Research/Lasagne")
import time

from os import listdir
from os.path import isfile, join
from lasagne.utils import floatX


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


def get_validation_labels(filename):
    """
    Extracts the label ids from given filename and returns as list.
    :param filename:
    :return:
    """
    labels = []
    with open(filename, "r") as f:
        for line in f:
            labels.append(int(line.strip()))

    one_hot_rep = np.zeros((len(labels), 1000), dtype=np.float32)
    for idx, label in enumerate(labels):
        one_hot_rep[idx, label-1] = 1.

    return one_hot_rep, labels


def get_image_id_mapping(filename):
    """
    Gets mapping from image ids to image categories(words)
    :param filename:
    :return:
    """
    id_to_category = {}
    with open(filename, "r") as f:
        # Disregard first line which just has column names
        _ = f.readline()
        for line in f:
            contents = line.split()
            id = int(contents[0])
            categories = " ".join(contents[2:])
            id_to_category[id] = categories

    return id_to_category


def scale_image(im):
    """
    Scales image to appropriate dimensions
    :param im:
    :return:
    """
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    return im


def compute_mean_image(data_dir):
    """
    Compute mean image of all images in given dir.
    :return:
    """
    mean_image = None
    files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

    # TODO: need to resize images from val set
    for idx, file in enumerate(files):
        img_arr = matplotlib.image.imread(data_dir+"/"+file)
        if len(img_arr.shape) == 2:
            h, w = img_arr.shape
            img_arr = img_arr.reshape(h, w, 1)

        converted_im = scale_image(img_arr)

        if idx == 0:
            mean_image = converted_im
        else:
            mean_image += converted_im

    mean_image /= len(files)

    return mean_image


def prep_image_data(filename, mean_image):
    """
    Preprocess data image from given dir with provided filename
    :param filename:
    :param mean_image:
    :return:
    """
    im = matplotlib.image.imread(filename)
    im = scale_image(im)

    rawim = np.copy(im).astype('uint8')

    im = im - mean_image[:, None, None]
    return rawim, floatX(im[np.newaxis])


def compute_accuracy_batch(val_fn, predict_fn, data_batch, labels_batch,
    img_id_mapping=None):
    """
    Compute accuracy of a data batch
    :param model:
    :param data_batch:
    :param labels_batch:
    :return:
    """
    _, acc = val_fn(data_batch, labels_batch)
    #predictions = predict_fn(data_batch)
    # print "Data batch: ", data_batch
    # print "Data batch shape:", data_batch.shape
    # print "Gold Labels batch: ", labels_batch
    # for label in labels_batch:
    #     print img_id_mapping[label]

    # print "Type gold labels: ", type(labels_batch)
    #predicted_labels = np.argmax(predictions, axis=1)
    # print "Predicted Labels: ", predicted_labels
    # for label in predicted_labels:
    #     print img_id_mapping[label]

    # print "Predicted Labels Shape: ", np.argmax(predictions, axis=1).shape
    # print "Predicted shape: ", predictions.shape
    return acc


def load_dataset_batch(data_dir, val_filename, batch_size, mean_image, lower_idx, upper_idx):
    """
    Given path to dir, containing image data, load an array of batch_size examples
    """
    files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

    # Reorder files so they are arranged numerically by img num
    img_nums = [int(f.split("_")[2].split(".")[0]) for f in files]
    file_to_img_num = {}
    for idx, f in enumerate(files):
        file_to_img_num[f] = img_nums[idx]
    sorted_files = sorted(file_to_img_num.items(), key=operator.itemgetter(1)) 
    files = [f[0] for f in sorted_files]

    # Get one-hot representations of validation dataset labels 
    one_hot_rep, labels = get_validation_labels(val_filename)

    # Take only subset of images/labels
    files = files[lower_idx:upper_idx]
    labels = labels[lower_idx:upper_idx]

    #print "Files: ", files
    #print "Labels: ", labels

    for start_idx in range(0, len(files) - batch_size + 1, batch_size):
        end_idx = start_idx + batch_size
        files_batch = files[start_idx:end_idx]
        labels_batch = labels[start_idx:end_idx]

        print "-"*100
        print files_batch
        print "\n"
        print labels_batch
        # TODO: Change hardcoding of dimensions
        data_batch = np.zeros((batch_size, 3, 224, 224))
        for idx, f in enumerate(files_batch):
            img = matplotlib.image.imread(data_dir+"/"+f)
            if len(img.shape) == 2:
                h, w = img.shape
                img = img.reshape(h, w, 1)

            # Convert to appropriate dimensions 3x224x224
            img = scale_image(img)

            # TODO: Check if necessary to normalize image here...
            img = img - mean_image[:, None, None]
            data_batch[idx, :, :, :] = img

        yield data_batch, labels_batch


# Barebones testing code

# get_image_id_mapping("/Users/mihaileric/Documents/CS231N/CS231N-FinalProject/datasets/parsedData.txt")
# get_validation_labels("/Users/mihaileric/Documents/"
#                     "CS231N/CS231N-FinalProject/datasets/ILSVRC2014_clsloc_validation_ground_truth.txt")
# start = time.time()
# print compute_mean_image("/Users/mihaileric/Documents/CS231N/CS231N-FinalProject/datasets/ILSVRC2012_img_val/")
# print "Time to compute mean image: ", str(time.time() - start)

# mean_image = np.array([[4], [4], [4]])
# prep_image_data("/Users/mihaileric/Documents/CS231N/CS231N-FinalProject/"
#                 "datasets/ILSVRC2012_img_val/ILSVRC2012_val_00050000.JPEG", mean_image)
