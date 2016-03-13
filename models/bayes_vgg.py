""" bayes_vgg 


	Runs on priors to give the adversarial examples.


"""
from fsgAdversarial import *
import densityestimation as de
import lasagne

def prepare_vgg():

    input_var = T.tensor4('inputs')
    model, classes, mean_image = load_data()
    net = build_model(input_var,batch_size=batch_size)
    lasagne.layers.set_all_param_values(net['prob'], model['param values'])

    return net
def run():
	net = prepare_vgg ()



if __name__ == __main__:
	run()