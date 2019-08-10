# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:05:37 2019

@author: Alvin AI
"""
import _pickle as pkl
import os
import numpy as np


# form reference1/2/3/4/5 .txt files
def make_ref(cappkl_path = './Flickr8k/cap_features/',
                      save_path = './Flickr8k/cap_features/ref'):
    """
    cappkl_path: the location of caption .pkl dataset.
    """
    def create_unseen_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)
    create_unseen_folder(save_path)
    
    for split in ['train', 'dev', 'test']:
        cappkl = cappkl_path + split + '.pkl'        
        with open(cappkl, 'rb') as f:
            caps = pkl.load(f)
        
        split_save_path = save_path + "/%s/" % split
        create_unseen_folder(split_save_path)
        
        print('Start making ref. for %s dataset' % split)
        
        for i in list(range(5)):
            ref_file = open(split_save_path+"reference%d.txt"%i, "w")
            for j in np.arange(0, len(caps), 5):
                idx = i + j
                ref_file.write("%s\n" % caps[idx][0])
            ref_file.close()
            print('Finish Processing reference %d' % int(i+1))
  
          
if __name__ == "__main__":
    make_ref(cappkl_path = './Flickr8k/cap_features_sap_noise_img_0.025/', save_path = './Flickr8k/cap_features_sap_noise_img_0.025/ref')