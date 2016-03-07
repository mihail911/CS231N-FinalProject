import sys

syns = sys.argv[1]

import pickle
import numpy as np
import matplotlib.pyplot as plt

ar = pickle.load(open('{0}'.format(syns)))
print ar.shape
#plt.hist(ar)
#plt.show()

