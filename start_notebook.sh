#/bin/bash 
# Usage: ./start_notebook.sh [virtualenv location] [port]

source .env/bin/activate
jupyter notebook --no-browser --port ${2-16385} 
