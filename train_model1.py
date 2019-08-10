# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:20:34 2019

@author: Alvin AI
"""

from capgen import train

def main():
    m_score = train(dim_word=512,  # word vector dimensionality
                  ctx_dim=512,  # context vector dimensionality
                  dim=1000,  # the number of LSTM units
                  attn_type="deterministic", # Soft atten
                  n_layers_att=2,  # number of layers used to compute the attention weights
                  n_layers_out=1,  # number of layers used to compute logit
                  n_layers_lstm=1,  # number of lstm layers
                  n_layers_init=1,  # number of layers to initialize LSTM at time 0
                  lstm_encoder=False,  # if True, run bidirectional LSTM on input units
                  prev2out=True,  # Feed previous word into logit
                  ctx2out=True,  # Feed attention weighted ctx into logit
                  alpha_entropy_c=0.002,  # hard attn param
                  RL_sumCost=True,  # hard attn param
                  semi_sampling_p=0.5,  # hard attn param
                  temperature=1.,  # hard attn param
                  patience=10,
                  max_epochs=5000,
                  dispFreq=1,
                  decay_c=0.,  # weight decay coeff
                  alpha_c=0.,  # doubly stochastic coeff
                  lrate=0.01,  # used only for SGD
                  selector=True,  # selector (see paper)
                  n_words=7632,  # vocab size
                  maxlen=100,  # maximum length of the description
                  optimizer='adam',
                  batch_size = 64,
                  valid_batch_size = 64,
                  saveto='model2.npz',  # relative path of saved model file
                  validFreq=1000,
                  saveFreq=100,  # save the parameters after every saveFreq updates
                  sampleFreq=500,  # generate some samples after every sampleFreq updates
                  dataset='flickr8k',
                  dictionary=None,  # word dictionary
                  use_dropout=False,  # setting this true turns on dropout at various points
                  use_dropout_lstm=False,  # dropout on lstm gates
                  reload_=False,
                  save_per_epoch=False,
                  dev_references='./Flickr8k/cap_features/ref/dev',
                  test_references='./Flickr8k/cap_features/ref/test',
                  use_metrics=True,
                  metric='Bleu_4') # this saves down the model every epoch
    print "Bleu_4: %.4f" % m_score
if __name__ == "__main__":
    main()