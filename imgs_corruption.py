# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:23:36 2019

@author: Alvin AI
"""

import os
import matplotlib.pyplot as plt

orig_img_path = './Flickr8k/test/'
targ_img_path = './Flickr8k/object_noise_img_dog/test/'

orig_img_fnames = os.listdir(orig_img_path)
targ_img_fnames = os.listdir(targ_img_path)

print orig_img_fnames[:5]==targ_img_fnames[:5] # validate whethter they are corresponding

same_nums = 0
diff_num_list = []
diff_pixel_val_sums = 0
for i in range(len(orig_img_fnames)):
    orig_img_array = plt.imread(orig_img_path + orig_img_fnames[i]).reshape(-1,224*224*3)
    targ_img_array = plt.imread(targ_img_path + targ_img_fnames[i]).reshape(-1,224*224*3)
    binary_list = orig_img_array == targ_img_array
    same_num = binary_list.sum()
    diff_num_list.append(orig_img_array.shape[1] - same_num)
    same_nums += same_num
    
    diff_pixel_val_sum = abs(orig_img_array - targ_img_array).sum()/diff_num_list[i]
    diff_pixel_val_sums += diff_pixel_val_sum
    
all_pixels_num = len(orig_img_fnames) * orig_img_array.shape[1]
averg_corrupt_pert = 1 - (same_nums/float(all_pixels_num))
averg_corrupt_extent = diff_pixel_val_sums / len(orig_img_fnames)

print 'The average corrupted proportion: %.4f' % averg_corrupt_pert
print 'The average corrupted extent:', averg_corrupt_extent

