import matplotlib.pyplot as plt
import sys
import theano
import theano.tensor as T
sys.path.append("/Users/mihaileric/Documents/Research/Lasagne")
import lasagne

from models.vgg19 import train_and_predict_funcs, build_model, load_weights
from util.util import show_net_weights, visualize_grid


def visualize_weights(layer_name, model):
    weights_layer = lasagne.layers.get_all_param_values(model[layer_name])
    filter_weights = weights_layer[-2].transpose(0, 2, 3, 1)
    #print filter_weights
    #print filter_weights.shape
    plt.imshow(visualize_grid(filter_weights, padding=1).astype('uint8'))
    plt.gca().axis('off')
    plt.gcf().set_size_inches(5, 5)
    plt.show()


if __name__ == "__main__":
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    _, _, classes, mean_image, values = load_weights("/Users/mihaileric/Documents/CS231N/CS231N-FinalProject/vgg19.pkl")
    print "Param values dtype:", values[0].dtype
    model = build_model(input_var)
    lasagne.layers.set_all_param_values(model['prob'], values)

    layer_name = "conv4_1"

    #visualize_weights(layer_name, model)