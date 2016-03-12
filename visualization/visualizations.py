import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import theano
import theano.tensor as T
sys.path.append("/Users/chris/Documents/cs231/project/CS231N-FinalProject/")
sys.path.append("/Users/mihaileric/Documents/Research/Lasagne")
import lasagne

from models.vgg19 import train_and_predict_funcs, build_model, load_weights
from util.util import show_net_weights, visualize_grid, get_label_to_synset_mapping, scale_image
from correspondence import find_nearest_trained

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


def convert_to_pixel_space(activations, filter_num, ubound=255.0):
    """
    Convert activations to a visually interpretable representation.
    :param activations:  Activations to convert to pixel space
    :param ubound: Image will be scaled to range [0, ubound]
    :return:
    """
    _, filters, H, W = activations.shape
    #rand_filter = np.random.choice(filters)


    filter_activations = activations[0, filter_num, :, :]
    low, high = np.min(filter_activations), np.max(filter_activations)
    converted_img = ubound * (filter_activations - low) / (high - low)

    return converted_img


def visualize_img(img):
    plt.imshow(img.astype('uint8'), interpolation='nearest')
    plt.colorbar()
    plt.gca().axis('off')
    plt.show()
     # Will hang both images for 20 seconds
    #plt.pause(20)


def compute_loss_grads_backprop_func(model, layer_name, input, filter_num):
    activations = lasagne.layers.get_output(model[layer_name], input)
    loss = activations[:, filter_num, :, :].mean()

    # Ensure that input is a symbolic rep of image
    grads = T.grad(loss, input)
    grads /= (T.sqrt(T.mean(T.square(grads))) + 1e-5)

    backprop_func = theano.function([input], [loss, grads])

    return backprop_func, loss, grads


def generate_optimal_img(backprop_func, input_img, num_iters=20, step_size=1.):
    for _ in range(num_iters):
        loss_value, grads_value = backprop_func(input_img)
        input_img += grads_value * step_size

    return input_img


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def reconstruct_features(model, layer_name, input_var, filename, filter_num):
    """
    Reconstruct features in input_img that led to max activation at given layer
    :param model:
    :param layer_name:
    :param input_var:
    :param input_img: Ensure 4d with (num_sample, num_filter, H, W)
    :return:
    """
    orig_img = plt.imread(filename)
    prep_orig = scale_image(orig_img)
    input_img = prep_orig[None, :, :, :].astype(np.float32)

    backprop_func, loss, grads = compute_loss_grads_backprop_func(model, layer_name, input_var, filter_num)

    loss, grads = backprop_func(input_img)

    print "Original image plotting"
    # Visualize img before ascent
    plt.figure(1)

    plt.imshow(np.transpose(input_img[0], (1, 2, 0)))
    plt.show(block=False)
    plt.pause(0.01)

    print "Input img shape: ", input_img.shape
    optimal_img = generate_optimal_img(backprop_func, input_img)

    print "Gradient ascent complete"

    plt.figure(2)
    # Visualize img after ascent
    deprocess_test = deprocess_image(optimal_img[0])
    plt.imshow(deprocess_test)
    plt.show()


def get_min_max_filter_idx(model, layer_name, img_orig, img_adv):
    """
    Compute L2 metric between activations for original/adversarial image.
    Get max/min activation metric across images.
    :param model:
    :param layer_name:
    :param img_orig:
    :param img_adv:
    :return:
    """
    img_orig_activation = get_activations_at_layer(model, layer_name, img_orig)
    img_adv_activation = get_activations_at_layer(model, layer_name, img_adv)

    sqr_diff = (img_orig_activation[0] - img_adv_activation[0])**2 # Dim (num_filter, H, W)
    filter_norm = np.sqrt(np.sum(sqr_diff, axis=(1, 2)))
    sorted_filter = np.argsort(filter_norm)

    # min, max
    return sorted_filter[0], sorted_filter[-1]


def visualize_max_filter_activations(model, layer_name, orig_filename, adv_filename):
     # Original image
    orig_img = plt.imread(orig_filename)
    prep_orig = scale_image(orig_img)
    prep_orig = prep_orig[None, :, :, :].astype(np.float32)

    adv_img = plt.imread(adv_filename)
    prep_adv = scale_image(adv_img)
    prep_adv = prep_adv[None, :, :, :].astype(np.float32)

    # Get min/max filter idx
    print "Computing max/min filters..."
    min_idx, max_idx = get_min_max_filter_idx(model, layer_name, prep_adv, prep_orig)

    print "Min idx {0}, Max idx {1}".format(str(min_idx), str(max_idx))

    print "Computing activations..."
    activations_orig = get_activations_at_layer(model, layer_name, prep_orig)
    activations_adv = get_activations_at_layer(model, layer_name, prep_adv)

    print "activations dim: ", activations_adv.shape

    print "Converting original image activations to pixel space..."
    converted_orig = convert_to_pixel_space(activations_orig, max_idx)
    plt.figure(1)

    visualize_img(converted_orig)

    print "Converting adversarial image activations to pixel space..."
    plt.figure(2)
    converted_adv = convert_to_pixel_space(activations_adv, max_idx)
    visualize_img(converted_adv)

    return min_idx, max_idx


if __name__ == "__main__":
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    _, _, classes, mean_image, values = load_weights("/Users/mihaileric/Documents/CS231N/CS231N-FinalProject/vgg19.pkl")

    model = build_model(input_var)
    lasagne.layers.set_all_param_values(model['prob'], values)

    layer_name = "conv1_2"

    # Original image
    # Change path to images as appropriate
    orig_filename = "/Users/mihaileric/Documents/CS231N/CS231N-FinalProject/datasets/nipun/orig_high_sea slug, nudibranch_0.94_jigsaw puzzle_0.97.png"
    adv_filename = "/Users/mihaileric/Documents/CS231N/CS231N-FinalProject/datasets/nipun/high_sea slug, nudibranch_0.94_jigsaw puzzle_0.97.png"

    #visualize_max_filter_activations(model, layer_name, orig_filename, adv_filename)

    # Expects inputs of (num_sample, channels, H, W) dim
    # Reconstruct features for original image
    reconstruct_features(model, layer_name, input_var, orig_filename, filter_num=28)


    # label_to_synset = get_label_to_synset_mapping\
    #     ("/Users/mihaileric/Documents/CS231N/CS231N-FinalProject/datasets/parsedData.txt")
    # img_synset = label_to_synset[img_label]
    # img, idx = find_nearest_trained(test_input_img, synset=img_synset)
    #
    # #converted_img = convert_to_pixel_space(activations)
