#/bin/bash 
# Usage: ./start_notebook.sh [virtualenv location] [port]

source ~/${1-deepenv}/bin/activate
jupyter notebook --no-browser --port ${2-16385} 
