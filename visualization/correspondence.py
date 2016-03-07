import sys
sys.path.append("/Users/chris/Documents/cs231/project/CS231N-FinalProject/")

import numpy as np 
import matplotlib.pyplot as plt

from models.diskreader import DiskReader


_metrics = { 'l2' : lambda img, arg: l2(img, arg),
			 'l1' : lambda img, arg: l1(img, arg)
			}


_reader = None

def find_nearest_trained (img, train_images=None, synset=None, metric='l2'):
	''' 
	Finds the nearest image to the parameter `img` in the train set of 
	images given by train_images, or the members of the synset
	given by `synset` (i.e. 'n04598582')

	Returns the nearest image under the metric defined by 'metric'.

	'''
	global _reader

	if train_images is None:
		if not _reader:
			_reader = DiskReader(prefix='../datasets/')
		assert synset is not None
	
		train_images = _reader.get(synset, delete=False)

	fn = _metrics[metric]	
	distances = fn(img, train_images)
	print distances

	ind = np.argmax(distances)
	return train_images[ind], ind

def l2(img, train):
	''' Finds l2 distance between image of size (C, H, W) and train set of size (N, C, H, W). '''
	subt = train - img
	return np.sum(subt*subt, axis=(1, 2, 3))

if __name__ == '__main__':
	A = np.random.random((10, 5, 6, 7))
	B = np.random.random((5, 6, 7))

	print l2(A, B)