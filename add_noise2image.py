# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 17:32:42 2019

@author: Alvin AI
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:27:58 2019

@author: Alvin AI
"""

from PIL import Image
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import os

def add_salt_and_pepper_noise(img_path, threshold, save_folder):
    """
    args:
        img_path: the path for a single image.
        threshod: threshold for noise and it cannot be larger than 0.5. If the 
                  threshold becomes larger, the noise becomes heavier. (0.025/0.1)
    """
    if threshold >= 0.5:
        print "The threshold value shouldn't be larger than 0.5."
    else:
        img = np.array(Image.open(img_path))
        rdn = np.random.uniform(0, 1, (224, 224))
        rdn = np.repeat(rdn[:, :, np.newaxis], 3, axis=2)
        img = np.where(rdn < threshold, 0, img)
        img = np.where(rdn > 1-threshold, 255, img)
        
        img = Image.fromarray(img)
        img_fname=img_path.split('/')[-1]
        img.save(save_folder+'/'+img_fname)


def add_block_noise(img_path, scale_ratio, save_folder):
    """
    args:
        img_path: the path for a single image.
        scale_ratio: this value controls the size ot block. If the ratio becomes
                     larger, the noise becomes smaller. (4/5)
        
    """
    img = np.array(Image.open(img_path))
    length = int(224 / scale_ratio)
    upper_limit = 224 - length
    # randomize the upper left point of the block
    x = np.random.randint(0, upper_limit, 1)[0]
    y = np.random.randint(0, upper_limit, 1)[0]
    img[x: x+length, y: y+length] = 0
    
    img = Image.fromarray(img)
    img_fname=img_path.split('/')[-1]
    img.save(save_folder+'/'+img_fname)
    
    
def add_object_noise(img_path, object_path, save_folder):
    img = Image.open(img_path)
    
    obj = Image.open(object_path)
    obj = obj.resize((int(224/4), int(224/4))) # also can be compared with block noise
    rotation = np.random.randint(0, 45, 1)[0] # rotation:0~45
    obj = obj.rotate(rotation)  
    
    upper_limit = 224 - int(224/4)
    # randomize the upper left point of the block
    x = np.random.randint(0, upper_limit, 1)[0]
    y = np.random.randint(0, upper_limit, 1)[0]    
    img.paste(obj, (x,y), obj)

    img = Image.fromarray(np.array(img))
    img_fname=img_path.split('/')[-1]
    img.save(save_folder+'/'+img_fname)    

def main(noise_type = 'salt_and_pepper_noise'):
    """
    arg:
        noist_type: the type of noiise.
                    - salt_and_pepper_noise
                    - block_noise
                    - object_noise
    """    

        
    for split in ['train', 'dev', 'test']:
        split_path = './Flickr8k/' + split
        img_fnames = os.listdir(split_path)
        img_fpaths = [split_path+'/'+fn for fn in img_fnames]
    
        save_folder = './Flickr8k/' + noise_type + '_0.025/' + split #!!remember change parameter in here
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    
        if noise_type == 'sap_noise_img':
            print 'threshold: ', 0.025
            for fpath in img_fpaths:
                add_salt_and_pepper_noise(fpath, 0.025, save_folder)
       
        elif noise_type == 'block_noise_img':
            print 'scale_ratio: ', 4
            for fpath in img_fpaths:
                add_block_noise(fpath, 4, save_folder)
        
        elif noise_type == 'object_noise_img_dog':
            print 'object: dog, resize: 224/4'
            # load object image
            object_path = './Flickr8k/dog.png'
   
            for fpath in img_fpaths: 
                add_object_noise(fpath, object_path, save_folder)
            
        print "Finish adding %s to %s dataset." % (noise_type, split)
    
if __name__ == '__main__':
    main(noise_type = 'sap_noise_img')
    #main(noise_type = 'block_noise')
    #main(noise_type = 'object_noise_img_dog')