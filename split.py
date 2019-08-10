# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 10:47:57 2019

@author: Alvin AI
"""
from PIL import Image
import os

def split_dataset():
    folder = './Flickr8k/Flickr8k_resized_Dataset/'
    splits = ['train', 'dev', 'test']
    #subsets_num = [3000, 500, 500] # only use the subset of the whole Flickr8k
    
    for i, split in enumerate(splits):
        split_folder = './Flickr8k/%s/' % split # save path
        if not os.path.exists(split_folder):
            os.makedirs(split_folder)
        # image file names in different splits
        split_files = './Flickr8k/Flickr8k_text/Flickr_8k.'+ split +'Images.txt' 
        
        print 'Start split %s dataset' % split
        
        
        with open(split_files, 'r') as f:
            for j, image_file in enumerate(f):
                if j % 100 == 0:
                    print 'Split image: %d' % j
                with Image.open(folder+image_file.strip()) as image:
                    image.save(os.path.join(split_folder, image_file.strip()), image.format)
            
if __name__ == '__main__':
    split_dataset()
                