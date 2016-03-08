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
url="http://www.image-net.org/download/synset"

params="wnid=$synset&username=$username&accesskey=$akey&release=latest&src=stanford"

curl -J -O \
    --data-urlencode wnid=$synset \
    --data-urlencode username=$username \
    --data-urlencode accesskey=$akey \
    --data           release=latest \
    --data           src=stanford \
    $url

mkdir -p "$datapath$synset"
tar xzf "$synset.tar" -C "$datapath$synset/"
rm "$synset.tar"

# Look through images, discard ones that cannot be the right size 
=======
echo "tar xzf $synset.tar -C $datapath$synset/"
tar xf "$synset.tar" -C "$datapath$synset/"
rm "$synset.tar"

# Look through images, discard ones that cannot be the right size 
