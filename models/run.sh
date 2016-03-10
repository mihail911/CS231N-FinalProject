#!/bin/bash
THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1' python $@
