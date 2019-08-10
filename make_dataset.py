# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:45:45 2019

@author: Alvin AI
"""

from vggnet import Vgg19
import tensorflow as tf
from scipy import ndimage
from collections import defaultdict
import _pickle as cPickle
import numpy as np
import hickle
import os 
import string
exclude = string.punctuation + "-"

# cite from https://github.com/yunjey/show-attend-and-tell
def extract_conv_feats(vgg_model_path = './imagenet-vgg-verydeep-19.mat', batch_size = 25):
    """
    Args:
        vgg_model_path: the patch to VGG model .mat file.
        batch_size: how many images are inputted into CNN model at the same time.
        
    Output:
        train/dev/test.feature.hkl: they contain huge feature vectors.
    """
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for split in ['train', 'dev', 'test']:
            feat_save_path = './Flickr8k/conv_features/%s_features.hkl' % split
            if not os.path.exists('./Flickr8k/conv_features/'):
                os.makedirs('./Flickr8k/conv_features/')
                
            img_path = './Flickr8k/' + split
            img_fnames = os.listdir(img_path)
            img_path = ['./Flickr8k/'+split+'/'+img_fname for img_fname in img_fnames]
            n_examples = len(img_fnames)
            
            all_feats = np.ndarray([n_examples, 196*512], dtype=np.float32)
            
            for start, end in zip(range(0, n_examples, batch_size),
                                  range(batch_size, n_examples + batch_size, batch_size)):
                image_batch_file = img_path[start:end]
                image_batch = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file))).astype(np.float32)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                feats = tf.reshape(feats, [-1, 14*14*512])
                all_feats[start:end, :] = feats.eval() # tensor to array 
                print "Processed %d %s features.." % (end, split)
                
            # use hickle to save huge feature vectors
            hickle.dump(all_feats, feat_save_path)
            print "Saved feature vectors to %s.." % (feat_save_path)

           
def extract_caps(lower =  True):
    """
    Args:
        lower: whether lowercase caption or not.
        
    Output:
        train/dev/test.feature.pkl: they contain caption information (caption, image_index).
        dictionary.pkl: the word dictionary of training data. (word, the number of occurence)
    """
    w_dict = defaultdict(int) # for vab. dictionary
    cap_file = './Flickr8k/Flickr8k_text/Flickr8k.token.txt'
    with open(cap_file, 'r') as f:
        caps = f.read().split('\n') 
        caps = caps[:-1]
        caps = [cap.split('\t') for cap in caps]
        # Flickr8k unlike COCO, each image has 5 image references
        for i in xrange(len(caps)):
            caps[i][0] = caps[i][0][:-2] # drop the '#<NUMBER>' symbol
        
    for split in ['train', 'dev', 'test']:
        data = dict()
        data['sents'] = []
        
        img_path = './Flickr8k/' + split
        img_fnames = os.listdir(img_path)
        
        cap_save_path = './Flickr8k/cap_features/%s.pkl' % split   
        if not os.path.exists('./Flickr8k/cap_features/'):
                    os.makedirs('./Flickr8k/cap_features/')
                
        for i, img_fnanme in enumerate(img_fnames):
            for j, cap in enumerate(caps):
                if cap[0] == img_fnanme:
                    if lower:
                        text = cap[1].lower()
                    text = ''.join(ch for ch in text if ch not in exclude)
                    text = text.strip()
                    data['sents'].append((text, i))
                    
                    # collect the vocabulary by only considering the training dataset
                    if split == 'train':
                        for w in text.split():
                            w_dict[w] += 1
        
        with open(cap_save_path, 'wb') as f:
            cPickle.dump(data['sents'], f, protocol=2)
        print "Saved %s caps data." %  split
        
    # Sort dictionary in descending order
    sorted_dict = sorted(w_dict, key=lambda x: w_dict[x], reverse=True)
    # Start at 2 because 0 and 1 are reserved
    numbered_dict = [(w, idx+2) for idx, w in enumerate(sorted_dict) if idx < 10000]
    print 'The length of the worddict:', len(list(numbered_dict))
    d_dict = dict(numbered_dict)    
    with open('./Flickr8k/cap_features/dictionary.pkl', 'wb') as f:
        cPickle.dump(d_dict, f, protocol=2)
    print "Saved word dictionary."
    print 'The length of worddict:', len(list(d_dict))
    
if __name__ == "__main__":
    extract_caps(lower =  True)
    extract_conv_feats(vgg_model_path = './imagenet-vgg-verydeep-19.mat', batch_size = 25)