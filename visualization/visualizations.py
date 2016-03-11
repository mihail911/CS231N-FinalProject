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
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')
    #plt.gcf().set_size_inches(5, 5)
    plt.show()


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

def reconstruct_features(model, layer_name, input_var, input_img, filter_num):
    """
    Reconstruct features in input_img that led to max activation at given layer
    :param model:
    :param layer_name:
    :param input_var:
    :param input_img: Ensure 4d with (num_sample, num_filter, H, W)
    :return:
    """
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

    num_filter, _, _ = img_orig_activation.shape

    sqr_diff = (img_orig_activation - img_adv_activation)**2 # Dim (num_filter, H, W)
    filter_norm = np.sqrt(np.sum(sqr_diff, axis=(1, 2)))
    sorted_filter = np.argsort(filter_norm)

    # min, max
    return sorted_filter[0], sorted_filter[-1]


def visualize_orig_adv_features(model, layer_name, input_var, img_orig, img_adv):
    # Visualize the features
    print "Getting min/max filters for adversarial/original images..."
    min_filter, max_filter = get_min_max_filter_idx(model, layer_name, img_orig, img_adv)

    # Min filter
    print "Reconstructing features for original image min filter..."
    reconstruct_features(model, layer_name, input_var, img_orig, min_filter)
    print "Reconstructing features for adversarial image min filter..."
    reconstruct_features(model, layer_name, input_var, img_adv, min_filter)

    # # Max filter
    print "Reconstructing features for original image max filter..."
    # reconstruct_features(model, layer_name, input_var, img_orig, max_filter)
    print "Reconstructing features for adversarial image max filter..."
    # reconstruct_features(model, layer_name, input_var, img_adv, max_filter)


if __name__ == "__main__":
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    _, _, classes, mean_image, values = load_weights("/Users/mihaileric/Documents/CS231N/CS231N-FinalProject/vgg19.pkl")

    model = build_model(input_var)
    lasagne.layers.set_all_param_values(model['prob'], values)

    layer_name = "conv5_1"
    test_input = np.random.random((1, 3, 224, 224)).astype(np.float32)*20 + 128
#    reconstruct_features(model, layer_name, input_var, test_input, filter_num=0)

    img_filename = "/Users/mihaileric/Documents/CS231N/CS231N-FinalProject/datasets/nipunresults/" \
                    "awsResults/advResults/vulture_0.08_beaver_0.31.png"
    test_img = plt.imread(img_filename)
    prep_img = scale_image(test_img)
    prep_img = prep_img[None, :, :, :].astype(np.float32)
    print "Img shape: ", prep_img.shape
    reconstruct_features(model, layer_name, input_var, prep_img, filter_num=0)



    # Code below relevant for other activation visualization method -- Not relevant to above
    # test_input_img = np.random.randn(3, 224, 224).astype(np.float32)
    # activations = get_activations_at_layer(model, layer_name, test_input)
    #
    # img_filename = "/Users/mihaileric/Documents/CS231N/CS231N-FinalProject/datasets/nipunresults/" \
    #                "awsResults/advResults/vulture_0.08_beaver_0.31.png"
    # img_label = "vulture"
    # img_file = None
    #
    #
    # label_to_synset = get_label_to_synset_mapping\
    #     ("/Users/mihaileric/Documents/CS231N/CS231N-FinalProject/datasets/parsedData.txt")
    # img_synset = label_to_synset[img_label]
    # img, idx = find_nearest_trained(test_input_img, synset=img_synset)
    #
    # #converted_img = convert_to_pixel_space(activations)
    #
    # # Swap channels back
    # #img = img[::-1, :, :]
    # # Swap axis order back to (224, 224, 3)
    # img = img.transpose(1,2,0)
    # visualize_img(img)

