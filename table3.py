import json
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import pprint
import nltk
import time
from nltk.tokenize.treebank import TreebankWordDetokenizer

from textbugger_utils import get_blackbox_classifier_score


def dwb_blackbox_SA(dataset, classifier_type, num_samples):
    
    data = pickle.load( open( "datasets/ADV/dwb_{}_adv.p".format(dataset), "rb" ) )
    
    pos_advs= data['pos'][0:num_samples]
    neg_advs = data['neg'][0:num_samples]

    num_pos = len(pos_advs)
    num_neg = len(neg_advs)
    total_advs = num_pos + num_neg
    successful_advs = 0

    ## Positive True Class
    start = time.time()
    for adv in pos_advs:
        y_pred = np.round(get_blackbox_classifier_score(classifier_type, adv),0)
        y_true = 1
        if (y_pred != y_true):
            successful_advs += 1

    ## Negative True Class
    for adv in neg_advs:
        y_pred = np.round(get_blackbox_classifier_score(classifier_type, adv),0)
        y_true = 0
        if (y_pred != y_true):
            successful_advs += 1   

    end = time.time()

    print("Total time: {} minutes".format((end-start)/60))
    print("Number of successful adversaries: {}".format(successful_advs))
    print("Total number of samples: {}".format(total_advs))
    print("{} success %: {}".format(classifier_type,np.round(successful_advs/total_advs,4)))
    


# dwb_blackbox_SA('RT', 'FB_fastText', 100)       # 38.5% # 3.730 min
# dwb_blackbox_SA('RT', 'Google_NLP', 100)        # 26.5% # 1.993 min
# dwb_blackbox_SA('RT', 'IBM_Watson', 100)        # 26.0% # 3.069 min
# dwb_blackbox_SA('RT', 'Microsoft_Azure', 100)   # 38.5% # 1.739 min
# dwb_blackbox_SA('RT', 'AWS_Comprehend', 100)    # 31.0% # 0.976 min




def get_original_bb_acc(dataset,classifier_type, num_samples):
    data = pickle.load( open( "datasets/{}/{}_tokens.p".format(dataset, dataset), "rb" ) )


    pos_docs = data['test']['pos'][0:num_samples]
    neg_docs = data['test']['neg'][0:num_samples]

    num_correct = 0
    num_total = len(pos_docs) + len(neg_docs)

    for tokens_list in pos_docs:
        sentence = TreebankWordDetokenizer().detokenize(tokens_list)
        y_pred_proba = get_blackbox_classifier_score(classifier_type, sentence)
        y_pred_class = np.round(y_pred_proba,0)
        if (y_pred_class == 1):
            num_correct += 1


    for tokens_list in neg_docs:
        sentence = TreebankWordDetokenizer().detokenize(tokens_list)
        y_pred_proba = get_blackbox_classifier_score(classifier_type, sentence)
        y_pred_class = np.round(y_pred_proba,0)
        if (y_pred_class == 1):
            num_correct += 1


    ori_acc = np.round((num_correct/num_total),3)


    print("Classifier: {}".format(classifier_type))
    print("Total number of samples: {}".format(num_total))
    print("Original Acc: {}".format(ori_acc))
    print()




# get_original_bb_acc('RT', 'FB_fastText', 100)       # 72.5%
# get_original_bb_acc('RT', 'Google_NLP', 100)        # 58.5%
# get_original_bb_acc('RT', 'IBM_Watson', 100)        # 62%
# get_original_bb_acc('RT', 'Microsoft_Azure', 100)   # 49%
# get_original_bb_acc('RT', 'AWS_Comprehend', 100)    # 69.5%

# get_original_bb_acc('IMDB', 'FB_fastText', 100)       # 55%
# get_original_bb_acc('IMDB', 'Google_NLP', 100)        # 56.5%
# get_original_bb_acc('IMDB', 'IBM_Watson', 100)        # 56.5%
# get_original_bb_acc('IMDB', 'Microsoft_Azure', 100)   # 49%
get_original_bb_acc('IMDB', 'AWS_Comprehend', 100)    # 69.5%

