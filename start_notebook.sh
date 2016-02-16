#/bin/bash 
# Usage: ./start_notebook.sh [virtualenv location] [port]

source .env/bin/activate
OMP_NUM_THREADS=4 jupyter notebook --no-browser --port ${1-16385} 
