import sys
import os
import time

import lasagne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

from os import listdir
from os.path import isfile, join
from util.util import scale_image, get_validation_labels
from models.vgg19 import build_model





# def train_and_predict_funcs(update="nesterov", regularization=0.0):
#     """
#     Create theano functions for computing loss, accuracy, etc. for given model
#     :param model:
#     :param update: Update parameter to use for training. Select from among
#                     "nesterov", "sgd", "rmsprop", etc.
#     :return:
#     """
#     input_var = T.tensor4('inputs')
#     target_var = T.ivector('targets')
#
#     model = build_model(input_var)["prob"]
#
#     # Create a loss expression for training, i.e., a scalar objective we want
#     # to minimize (for our multi-class problem, it is the cross-entropy loss):
#     prediction = lasagne.layers.get_output(model)
#     loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
#     loss = loss.mean()
#     # TODO: Add regularization to the loss
#
#     params = lasagne.layers.get_all_params(model, trainable=True)
#     if update == "nesterov":
#         updates = lasagne.updates.nesterov_momentum(
#             loss, params, learning_rate=0.01, momentum=0.9)
#     else:
#         pass
#
#     # Create a loss expression for validation/testing. The crucial difference
#     # here is that we do a deterministic forward pass through the network,
#     # disabling dropout layers.
#     test_prediction = lasagne.layers.get_output(model, deterministic=True)
#     test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
#                                                             target_var)
#     test_loss = test_loss.mean()
#
#     test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
#                       dtype=theano.config.floatX)
#
#     # Compile a function performing a training step on a mini-batch (by giving
#     # the updates dictionary) and returning the corresponding training loss:
#     train_fn = theano.function([input_var, target_var], loss, updates=updates)
#
#     # Compile a second function computing the validation loss and accuracy:
#     val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
#
#     # theano function giving output label for given input
#     predict_fn = theano.function([input_var, target_var], test_prediction)
#
#     return train_fn, val_fn, predict_fn


def train(num_epochs=10):
    batch_size = 500
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    model = None
    train_fn, val_fn, predict_fn = train_and_predict_funcs(model)

    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))


def load_dataset():
    # NOTE: Need to implement this function in order for compute_accuracy
    # to work
    pass


# TODO: Refactor this to make use of compute_accuracy_batch
# def compute_accuracy(model, test=False):
#     """
#
#     :param model: Model to compute accuracy with respect to
#     :param test:  Whether or not to compute accuracy on test dataset (only use when
#                     performing final run)
#     :return: Computed accuracy values
#     """
#     _, val_fn, predict_fn = train_and_predict_funcs(model)
#     X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
#
#     _, train_acc = val_fn(X_train, y_train)
#     _, dev_acc = val_fn(X_val, y_val)
#
#     test_acc = None
#     if test:
#         _, test_acc = val_fn(X_test, y_test)
#         print "Train accuracy: {0}, Dev accuracy: {1}, Test accuracy: {2}".\
#             format(train_acc, dev_acc, test_acc)
#     else:
#         print "Train accuracy: {0}, Dev accuracy: {1}".\
#             format(train_acc, dev_acc)
#
#     return train_acc, dev_acc, test_acc