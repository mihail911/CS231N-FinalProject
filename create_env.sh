#/bin/bash

virtualenv --system-site-packages .env
source .env/bin/activate

# Dependency issue in scikit-image 0.11.3 requires numpy first
# https://github.com/scikit-learn/scikit-learn/issues/4164
HAS_NUMPY=`python -c 'import numpy'`

if [ $? -eq 1 ] 
then
   echo "numpy not found, installing..."
   pip install numpy
fi

HAS_SCIPY=`python -c 'import scipy'`

if [ $? -eq 1 ] 
then
   echo "scipy not found, installing..."
   pip install scipy==0.17.0
fi

pip install -r requirements.txt
# force install of jupyter
pip install -I jupyter==1.0.0
