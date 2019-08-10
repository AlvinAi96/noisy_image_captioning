# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:17:59 2019

@author: Alvin AI
"""

import metrics
import os
import argparse

# helper function to calculate and retrieve the metric score of
# generated sentences. Defaults to returning Bleu_1, controlled by the
# options['metric'] parameter
def check_metrics(hyps, refs):
    # This block of code is extremely ugly
    hyps = hyps + ".txt"
    parser = argparse.ArgumentParser()
    refs_list = os.listdir(refs)
    refs_list = ["%s/%s" % (refs, x) for x in refs_list]
    parser.add_argument("refs", type=argparse.FileType('r'), nargs="+") # nargs="+"：all command-line args present are gathered into a list. 
    parser.add_argument("hyps", type=argparse.FileType('r'))
    refs_list.append(hyps)
    args = parser.parse_args(refs_list)

    references, hypotheses = metrics.load_textfiles(args.refs, args.hyps)
    scores = metrics.score(references, hypotheses)
    for k in list(scores):
        scores[k] = scores[k]*100
    return scores # shift scores from 0-1 to 0-100

if __name__ == '__main__':
    #train_scores = check_metrics('./outputs/model2_generate.train', './Flickr8k/cap_features/ref/train')
    #dev_scores = check_metrics('./Flickr8k/sap_noise_img_0.1_generate_outputs/generate.dev', './Flickr8k/cap_features_sap_noise_img_0.1/ref/dev')
    test_scores = check_metrics('./Flickr8k/sap_noise_img_0.025_generate_outputs/generate.test', './Flickr8k/cap_features_sap_noise_img_0.025/ref/test')
    #print 'train_scores:', train_scores
    #print 'dev_scores:', dev_scores
    print 'test_scores:', test_scores