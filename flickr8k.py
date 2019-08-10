import cPickle as pkl
import gzip
import os
import sys
import time
import hickle
import numpy


def prepare_data(caps, features, worddict, maxlen=None, n_words=7632, zero_pad=False):
    # x: a list of sentences
    seqs = []
    feat_list = []
    for cc in caps:
        seqs.append([worddict[w] if w in list(worddict) and worddict[w] < n_words else 1 for w in cc[0].split()])
        feat_list.append(features[cc[1]])

    lengths = [len(s) for s in seqs]

    if maxlen != None and numpy.max(lengths) >= maxlen:
        new_seqs = []
        new_feat_list = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, feat_list):
            if l < maxlen:
                new_seqs.append(s)
                new_feat_list.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        feat_list = new_feat_list
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    y = numpy.zeros((len(feat_list), 14*14, 512)).astype('float32')
    for idx, ff in enumerate(feat_list):
        #ff = csr_matrix(ff) # for .todense() function
        #y[idx,:] = numpy.array(ff.todense())
        y[idx,:] = ff.reshape(14*14, 512)
    if zero_pad:
        y_pad = numpy.zeros((y.shape[0], y.shape[1]+1, y.shape[2])).astype('float32')
        y_pad[:,:-1,:] = y
        y = y_pad

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.

    return x, x_mask, y


#def load_data(load_train=True, load_dev=True, load_test=True,        
              #img_feats_path='./Flickr8k/conv_features/',
              #cap_path='./Flickr8k/cap_features/'):
def load_data(load_train=True, load_dev=True, load_test=True,        
              img_feats_path='./Flickr8k/conv_features_object_noise_img_dog/',
              cap_path='./Flickr8k/cap_features_object_noise_img_dog/'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data'

    if load_train:
        train_cap = pkl.load(open(cap_path+'train.pkl', 'rb'))
        train_feat = hickle.load(img_feats_path+'train_features.hkl')
        train = (train_cap, train_feat)
    else:
        train = None
    if load_test:
        test_cap = pkl.load(open(cap_path+'test.pkl', 'rb'))
        test_feat = hickle.load(img_feats_path+'test_features.hkl')
        test = (test_cap, test_feat)
    else:
        test = None
    if load_dev:
        dev_cap = pkl.load(open(cap_path+'dev.pkl', 'rb'))
        dev_feat = hickle.load(img_feats_path+'dev_features.hkl')
        valid = (dev_cap, dev_feat)
    else:
        valid = None


    with open(cap_path+'dictionary.pkl', 'rb') as f:
        worddict = pkl.load(f)

    return train, valid, test, worddict
