#!/bin/bash
# set up data directory

access_key=$1
cd /mnt/
mkdir -p data
sudo chown -R ubuntu:ubuntu data
# theano flags
export THEANO_FLAGS='floatX=float32, device=gpu0'

# set up access key
cd ~/CS231N-FinalProject/util/
echo access_key >> ACCESS_KEY
cd ..

# get weights
mkdir -p weights
cd weights
wget https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl
