
Getting Started on AWS
---------------------------------------
Follow the instructions in the CS231N tutorial.

* If downloading training set images, make sure to put 
  the access key in ACCESS_KEY in the util/ folder next to imgdownloader.sh. 
  Change the flags appropriately.




And you should be set.

Tips for running on AWS:

Set the theano flags in a script like this:

THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=0.9' python fsgAdversarial.py
