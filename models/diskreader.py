import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess as sp
import shlex
import io
import skimage.transform

import multiprocessing as mp
import os
from lasagne.utils import floatX

class DiskReader (object):
    ''' 
    Provides images off of disk over a socket, and stores
    data locally with a synset.
    
    One thread servers as the base server to bind to, and the other does
    computation in the background.
    '''
    def __init__(self):
        self.activeQueues = {}  
        self.data = {}
    def startRequest (self, synset):
        ''' Begins a request to asynchronously fetch the images for a synset. 
            Returns nothing.
        '''
        if synset in self.activeQueues:
            raise Exception # CANT DO THAT

        print 'Starting process'
        q = mp.Queue()
        proc = mp.Process(target=self.processImages, args=((synset, q), ))
        self.activeQueues[synset] = (q, proc)
        proc.start()
    
    def remove(self, synset):
        ''' Removes a class of images from the map to improve memory usage.'''
        if synset in self.data:
            self.data[synset] = None

    def get(self, synset, delete=True):
        ''' Performs the asynchronous get() on the synset data, and wipes the process
            once it is done.

            Returns samples
        '''

        if synset in self.data:
            return self.data[synset]
        if synset not in self.activeQueues:
            self.startRequest(synset)
        
        q, proc = self.activeQueues[synset]
        result = q.get()
	print "images shape: ", result.shape
        if not delete:
            self.data[synset] = result
        print "processing finished"
        proc.join()
        self.activeQueues[synset] = None
        # null out the active queues

        return result

    def ensureDataExists (self, prefix, synset):
        ''' Downloads the synset data and unpacks it if it doesn't exist already. '''
        if os.path.exists(prefix):
            return 
        else:
            print "Downloading packaged synset..."
            cmd = shlex.split("../util/./imgdownloader.sh {0}".format(synset))
            sp.call(cmd)

    def processImages (self, args):
        synset, q = args

        ''' Processes the images from a directory on disk '''
        count = 0
        prefix = '/mnt/data/{0}/'.format(synset)

        self.ensureDataExists(prefix, synset)

        files = os.listdir(prefix)
        N = len(files)
        images = np.zeros ((N, 3, 224, 224))
        print "processing in progress..."
        chopped_off = 0
	for i, f in enumerate(files):
            im = plt.imread (prefix + f)

            sh = im.shape
            if len(sh) <= 2:
                im = im[:, :, None]
	    elif im.shape[2] == 4: # wtf is this image: skip
        	chopped_off += 1
		continue    
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
            count += 1
            if count % 10 == 0:
                print count    
            # Convert to BGR
            images[i, :, :, :] = im[::-1, :, :]
        images = images[:i - chopped_off, :, :, :]
	print images.shape
	
        q.put( floatX(images) )

#-------------------------------
if __name__ == '__main__':
    
    bts = DiskReader()
    bts.get('n04598582')
