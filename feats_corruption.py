# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:35:01 2019

@author: Alvin AI

This .py is for compute the distortion of image features corrupted by image noise
"""

import hickle
import numpy as np

orig_imgfeats_path='./Flickr8k/conv_features/' 
orig_test_feat = hickle.load(orig_imgfeats_path+'test_features.hkl') # original image features of test dataset

targ_imgfeats_path='./Flickr8k/conv_features_block_noise_img_4/'  
targ_test_feat = hickle.load(targ_imgfeats_path+'test_features.hkl') # target image features of test dataset

same_nums = 0 # the number of same-value pixels in all images
diff_num_list = [] # the list for the number of different-value pixels of each image
for i in range(len(orig_test_feat)):
    binary_list = orig_test_feat[i] == targ_test_feat[i]
    same_num = binary_list.sum()
    diff_num_list.append(len(orig_test_feat[0])-same_num)
    same_nums += same_num

all_pixels_num = len(orig_test_feat) * len(orig_test_feat[0]) # the total pixel amount of all images
averg_corrupt_pert = 1 - (same_nums/float(all_pixels_num))

print 'The average corrupted proportion: %.4f' % averg_corrupt_pert

diff_pixel_val_sums = 0
for j in range(len(orig_test_feat)):
    diff_pixel_val_sum = abs(orig_test_feat[j] - targ_test_feat[j]).sum()/diff_num_list[j]
    diff_pixel_val_sums += diff_pixel_val_sum
    
averg_corrupt_extent = diff_pixel_val_sums / len(orig_test_feat)
    
print 'The average corrupted extent:', averg_corrupt_extent

