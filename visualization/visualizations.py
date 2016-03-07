import matplotlib.pyplot as plt
import numpy as np
import sys
import theano
import theano.tensor as T
sys.path.append("/Users/chris/Documents/cs231/project/CS231N-FinalProject/")
import lasagne

from models.vgg19 import train_and_predict_funcs, build_model, load_weights
from util.util import show_net_weights, visualize_grid


def visualize_weights(layer_name, model):
    weights_layer = lasagne.layers.get_all_param_values(model[layer_name])
    filter_weights = weights_layer[-2].transpose(0, 2, 3, 1)
    plt.imshow(visualize_grid(filter_weights, padding=1).astype('uint8'))
    plt.gca().axis('off')
    plt.gcf().set_size_inches(5, 5)
    plt.show()


def get_activations_at_layer(model, layer_name, input):
    """
    Get activations at a given layer of model specified by layer_name
    :param model:
    :param layer_name:
    :param input:
    :return:
    """
    activations = lasagne.layers.get_output(model[layer_name], input)
    return activations.eval()


def convert_to_pixel_space(activations, ubound=255.0):
    """
    Convert activations to a visually interpretable representation.
    :param activations:  Activations to convert to pixel space
    :param ubound: Image will be scaled to range [0, ubound]
    :return:
    """
    _, filters, H, W = activations.shape
    rand_filter = np.random.choice(filters)

    print "Rand_filter: ", rand_filter

    filter_activations = activations[0, rand_filter, :, :]
    low, high = np.min(filter_activations), np.max(filter_activations)
    converted_img = ubound * (filter_activations - low) / (high - low)

    return converted_img


def visualize_img(img):
    print "img shape: {0}", img.shape
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')
    plt.gcf().set_size_inches(5, 5)
    plt.show()



if __name__ == "__main__":
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    _, _, classes, mean_image, values = load_weights("/Users/mihaileric/Documents/CS231N/CS231N-FinalProject/vgg19.pkl")

    # Save Idx to Label Mapping for pretrained weights
    # with open("pretrained_labels.txt", "w") as f:
    #     for idx, label in enumerate(classes):
    #         f.write("Label {0}: {1}\n".format(str(idx+1), str(label)))

    model = build_model(input_var)
    lasagne.layers.set_all_param_values(model['prob'], values)

    layer_name = "conv1_1"
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    activations = get_activations_at_layer(model, layer_name, test_input)

    converted_img = convert_to_pixel_space(activations)
    visualize_img(converted_img)

