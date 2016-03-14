""" bayes_vgg 


	Runs on priors to give the adversarial examples.


"""
from fsgAdversarial import *
import densityestimation as de
import lasagne
import time
import glob
import re
import csv

def parse_filename (name):
    pattern = 'high_(.*)_(\d\.\d\d)_(.*)_(\d\.\d\d)'
    print name
    match = re.search(pattern, name)
    if not match:
	print "no match found for ", name
	return "UNKK", 0.0, "NOWN", 0.0
    
    return match.groups()

def read_parsedData():
	syn_to_label = {}
	line_no = 0
	with open('../datasets/crap.txt', 'rb') as f:
	    reader = csv.reader (f, delimiter='\t') 
	    for row in reader:
		if line_no >= 1000: break    
		syn_to_label[row[0]] = row[1]	  
		line_no += 1
	print "stl len: ", len(syn_to_label)
	return syn_to_label

	
def prepare_vgg(batch_size=96):

    input_var = T.tensor4('inputs')
    model, classes, mean_image = load_data()
    print "mean image: ", mean_image
    net = build_model(input_var,batch_size=batch_size)
    lasagne.layers.set_all_param_values(net['prob'], model['param values'])

    return net, mean_image, model['param values'], classes

def get_synset_map ():
	''' Derive mapping from synsets to class labels '''
	syn_map = {}
	i = 0	
	with open('../util/synsets') as f:
	    
	    for line in f:
		if i >= 1000: break
		syn_map[i] = line.rstrip()
		syn_map[ syn_map[i] ] = i
		i += 1
	return syn_map
	
def load_images (mean_image):
	''' Loads the original and adversarial images from disk. '''
    	N = 1000
	files = glob.glob('../datasets/fsgResults/orig*')[:N]
	
	orig_images = np.zeros ((N, 3, 224, 224))
	adv_images = np.zeros ((N, 3, 224, 224))

	name_pairs = []

	for i, f in enumerate(files): 
	    print f
            _, orig_images[i, :, :, :] = prep_image (f, mean_image)
	    adv_f = f.replace ('orig_', '')
	    print adv_f
	    name_pairs.append((f, adv_f))
	    _, adv_images[i, :, :, :] = prep_image (adv_f, mean_image) 	
	
	return floatX(orig_images), floatX(adv_images), name_pairs
 

def prep_fully_connected(input_var, param_values, batch_size =None):
    net = {}
    net['input'] = InputLayer((batch_size, 4096),input_var=input_var)
    net['fc6_dropout'] = DropoutLayer(net['input'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    lasagne.layers.set_all_param_values(net['prob'], param_values)   
    return net


def forward (net, images, batch_size=96, all_the_way=False):
	''' Duplicate of method in bootstrapper.py '''
        N, C, H, W = images.shape
	if all_the_way:
	    features = np.zeros((N, 1000))
	else:
       	    features = np.zeros ((N, 4096))

        for n in xrange(0, N, batch_size):
	    t0 = time.time()
            curSet = images[n:n+batch_size, :, :, :]
	    print "IMAGES: ", images[:, 0, :3, :3]
	    print "beginning run on {0} : {1}".format(n, n+batch_size)
            if all_the_way:
		features[n:n+batch_size, :] = np.array(lasagne.layers.get_output(
                                                    net['prob'], curSet, deterministic=True).eval(),
                                                    dtype=np.float32)

	    else:	
	        features[n:n+batch_size, :] = np.array(lasagne.layers.get_output(
                                                    net['fc6'], curSet, deterministic=True).eval(), 
                                                    dtype=np.float32)

		
	    t1 = time.time()
	    print "Finished batch {0} - {1} in {2:.3f} seconds".format(n, n+batch_size, t1 - t0)
	return floatX(features)

def run():
	
	net, mean_img, param_values, classes = prepare_vgg ()

	imnet_to_classes = {i : classes[i] for i in xrange(len(classes))}	
	classes_to_imnet = {classes[i] : i+1 for i in xrange(len(classes))}	
        
	stl = read_parsedData()
	print "Net prepared"
	input_var = T.matrix('inputs')
	fc_net = prep_fully_connected (input_var, param_values[-4:])
	print "fully connected layer finished"
	
	syn_map = get_synset_map()
	
	temp_map = {}
	for syn in stl:
	   
	   label = stl[syn]
	   ind = classes_to_imnet[label]
	   	   
	   temp_map[ind] = syn
	temp_map[135] = 'n02486261'
	
	print temp_map, len(temp_map)	
	N = 1000
	synsets = [temp_map[i] for i in xrange(1, len(temp_map) + 1) ]

	print "loaded synsets: ", len(synsets)
	orig_images, adv_images, name_pairs = load_images (mean_img)
	print "loaded adv examples"
	
	for j in xrange(N):	
            orig_label, orig_conf, adv_label, adv_conf = parse_filename(name_pairs[j][0])
	    print orig_label, orig_conf, adv_label, adv_conf

	adv_features = forward (net, adv_images, all_the_way = False)
	
	adv_prob = forward (net, adv_images, all_the_way=True)

	adv_density = de.form_density_priors (synsets, vgg_out=adv_features).T
	
	print adv_density.shape
        print adv_prob.shape		
	print name_pairs[0]	
	for i, syn in enumerate(synsets):
		pass
		# print syn, classes[i],  adv_density[:, i], adv_prob[:, i]	
	
	posterior = adv_prob * adv_density
	partition = np.sum(posterior, axis=1)[:, None] 
	posterior /= partition
	
	#print name_pairs
	for i, syn in enumerate(synsets):
	    pass
	#	print "Label: before | after:", syn, posterior[:, i], adv_prob[:, i]
  
	N = adv_prob.shape[0]
	
	# print to file
	orig_f = open('original_adv_results.txt', 'w+')
	adv_f = open('new_adv_results.txt', 'w+')
	for ex in xrange(N):
	    orig_label, orig_conf, adv_label, adv_conf = parse_filename(name_pairs[ex][0])
            orig_f.write("\t".join([orig_label, str(orig_conf), adv_label, str(adv_conf)]) + "\n")
	   
	    old_adv_ind = np.argmax( adv_prob[ex, :] )
	    old_label_conf = posterior[ex, old_adv_ind]
	    	    

	    new_ind = np.argmax (posterior[ex, :])
	    new_conf = posterior[ex, new_ind]
            adv_f.write('\t'.join([adv_label, str(old_label_conf), classes[new_ind], str(new_conf)]) + '\n')      
	
	orig_f.close()
	adv_f.close()
	print "done" 
     
if __name__ =='__main__':
	run()
	
