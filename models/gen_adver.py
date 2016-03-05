# coding: utf-8
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
		self.built_backnet = None
	

	def build_network(self,net,input_var):
		if net is 'vgg19':
			network = InputLayer(shape=(self.num_images,3, 224, 224), input_var=input_var)
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



	def adExample(self,X,y,weights,network,input_var):
		target_var = T.ivector('targets')
                
		if self.built_backnet is None:
			self.built_backnet = self.build_network('vgg19', input_var)
		network = self.built_backnet
		# YES, this does clobber the namespace
		# TODO: fix
	
		prediction = lasagne.layers.get_output(network)

		if self.loss is 'softmax':
			loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
		if self.loss is 'svm':
			loss = lasagne.objectives.multiclass_hinge_loss(prediction, target_var)

		loss = loss.mean()
		params = lasagne.layers.get_all_params(network, trainable=True)

		lasagne.layers.set_all_param_values(network, weights)
		Xnew = np.zeros((self.num_images,3,224,224))
		Xnew[:,:,:,:] = X

		grad = T.grad(loss, input_var)
		final_examples = X + self.eps*T.sgn(grad)
		func1 = theano.function([input_var,target_var], final_examples, allow_input_downcast=True)
		
		print "xnew shape", Xnew.shape
		print "y shape", y.shape
		result = func1(Xnew,y)
		return result


		






