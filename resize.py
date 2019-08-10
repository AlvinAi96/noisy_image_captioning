# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 10:41:47 2019

@author: Alvin AI
"""

from PIL import Image
import os


def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image

def main():
        folder = './Flickr8k/Flickr8k_Dataset' # dataset path
        resized_folder = './Flickr8k/Flickr8k_resized_Dataset/' # save path
        if not os.path.exists(resized_folder):
            os.makedirs(resized_folder)
        print 'Start resizing all the images.'
        image_files = os.listdir(folder)
        num_images = len(image_files)
        for i, image_file in enumerate(image_files):
            with open(os.path.join(folder, image_file), 'r+b') as f:
                with Image.open(f) as image:
                    image = resize_image(image)
                    image.save(os.path.join(resized_folder, image_file), image.format)
            if i % 100 == 0:
                print 'Resized images: %d/%d' % (i, num_images)
              
            
if __name__ == '__main__':
    main()