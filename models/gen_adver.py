#!/usr/bin/python
import sys
import re
import collections
import subprocess
import shlex
import numpy as np
import theano
import theano.tensor as T
from theano import pp

import matplotlib.pyplot as plt
import pickle
import urllib
import io
import skimage.transform

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.conv import Conv2DLayer as ConvLayer
from lasagne.layers.pool import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from lasagne.nonlinearities import softmax


class FastGradient(object):

	def __init__(self, num_images = 1,input_dim=(3,224,224), eps=0.0007, loss='softmax' ,dtype=np.float64):
		self.num_images = num_images
		self.input_dim = input_dim
		self.eps = eps
		self.dtype = dtype
		self.loss = loss

	def build_network(self,net,input_var):
		if net is 'vgg19':
			network = InputLayer(shape=(self.num_images, 3, 224, 224), input_var=input_var)
			network = ConvLayer(network, 64, 3, pad=1)
			network = ConvLayer(network, 64, 3, pad=1)
			network = PoolLayer(network, 2)
			network = ConvLayer(
			    network, 128, 3, pad=1)
			network = ConvLayer(
			    network, 128, 3, pad=1)
			network = PoolLayer(network, 2)
			network = ConvLayer(
			    network, 256, 3, pad=1)
			network = ConvLayer(
			    network, 256, 3, pad=1)
			network = ConvLayer(
			    network, 256, 3, pad=1)
			network = ConvLayer(
			    network, 256, 3, pad=1)
			network = PoolLayer(network, 2)
			network = ConvLayer(
			    network, 512, 3, pad=1)
			network = ConvLayer(
			    network, 512, 3, pad=1)
			network = ConvLayer(
			    network, 512, 3, pad=1)
			network = ConvLayer(
			    network, 512, 3, pad=1)
			network = PoolLayer(network, 2)
			network = ConvLayer(
			    network, 512, 3, pad=1)
			network = ConvLayer(
			  	network, 512, 3, pad=1)
			network = ConvLayer(
			    network, 512, 3, pad=1)
			network = ConvLayer(
			    network, 512, 3, pad=1)
			network = PoolLayer(network, 2)
			network = DenseLayer(network, num_units=4096)
			network = DropoutLayer(network, p=0.5)
			network = DenseLayer(network, num_units=4096)
			network = DropoutLayer(network, p=0.5)
			network = DenseLayer(
			    network, num_units=1000, nonlinearity=None)
			network = NonlinearityLayer(network, softmax)
		else:
			print "Input valid Model"

		return network



	def gradient(self,net='vgg19'):
		input_var = T.tensor4('inputs')
		target_var = T.ivector('targets')
		network = self.build_network(net,input_var)

		prediction = lasagne.layers.get_output(network)
		if self.loss is 'softmax':
			loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
		if self.loss is 'svm':
			loss = lasagne.objectives.multiclass_hinge_loss(prediction, target_var)

		loss = loss.mean()

		params = lasagne.layers.get_all_params(network, trainable=True)
		grad = T.grad(loss, input_var)

		return network,grad


	def adver_examples(self, network,grad, X, y, weights):
		lasagne.layers.set_all_param_values(network, weights)
		f = theano.function([X], grad)
		result = f(X,y)
		final_examples = X + self.eps*np.sign(result)
		return final_examples






