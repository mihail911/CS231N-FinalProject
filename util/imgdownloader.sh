#!/bin/bash

synset=${1-n01440764}
akey=$(cat ../util/ACCESS_KEY)
username=cjbillov

echo $username
echo $synset
echo $akey
# in util/, want datasets/
datapath="../datasets/"

# Download synset tar
url="http://www.image-net.org/download/synset?wnid=$synset&username=$username&accesskey=$akey&release=latest&src=stanford"

curl -J -O $url
mkdir -p "$datapath$synset"
tar xzf "$synset.tar" -C "$datapath$synset/"
rm "$synset.tar"

# Look through images, discard ones that cannot be the right size 