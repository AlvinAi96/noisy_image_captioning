### THE EFFECT OF IMAGE NOISE ON IMAGE CAPTION GENERATION
#### Author: Hongfeng Ai
#### Programming Software: Python 2.7
#### Dependencies: 
tensorflow 1.14.0  
scipy 1.2.1  
Pillow 6.0.0  
numpy 1.15.0  
theano 1.0.3  
hickle 3.4.3  
matplotlib 2.2.4  
scikit-learn 0.20.3  
#### Explanation:
split.py:                      |Spliting Flickr8k dataset into train/val/test dataset.  
resize.py:                     |Resize the size of images to 224 x 224.  
vggnet.py:                     |Construct VGG19 architecture.  
make_dataset.py:               Extract image features and format the textual captions in a proper way.  
make_ref.py:                   Form reference1/2/3/4/5 .txt files.  
flickr8k/flickr30k/coco.py:    Prepare and load the data.  
optimizers.py:                 Optimizers (e.g., RMSprop, Adam and so on).  
metrics.py:                    Evaluation metrics - BLEU, ROUGE, METEOR, and CIDER.  
homogeneous_data.py:           Ensure each gradient descent use the captions that share the same length.  
capgen.py:                     The attention-based image captioning model.  
train_model1.py:               Train the model.  
generate_caps.py:              Generate captions by using the fitted model.  
evaluate_metrics.py:           Evaluate the generated captions.  
alpha_visualization.ipynb:     Visualize the attention distribution.  
add_noise2image.py:            Add image noise on images.  
make_dataset_fornoiseimage.py: Make the noisy input for the model.  
image_corruption.py:           Image-level noise quantification.
feat_corruption.py:            Feature-level noise quantification.

To successfully evaluate the model, please visit [4] download the file - pycocoevalcap and visit [5] download the pretrained model file - imagenet-vgg-verydeep-19.mat.
#### References:  
[1] Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio. To appear ICML (2015)  
[2] https://github.com/elliottd/satyrid  
[3] https://github.com/yunjey/show-attend-and-tell  
[4] https://github.com/tylin/coco-caption  
[5] http://www.vlfeat.org/matconvnet/pretrained/
