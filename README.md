
Getting Started (from corn machines)
---------------------------------------

Your shell environment should be bash.

* Run ./create_env.sh to set up your initial virtual environment.
* Run ./start_notebook.sh to set up the jupyter notebook environment.


On your local machine, set up port forwarding:
* ssh -L [remote port]localhost:[local port] [username]@corn[01-39].stanford.edu

And you should be set.

Tips for running on AWS:

Set the theano flags in a script like this:

THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=0.9' python fsgAdversarial.py
