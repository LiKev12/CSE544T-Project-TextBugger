import json
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import pprint
import nltk
import time

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
    


dwb_blackbox_SA('RT','FB_fastText', 100)        # 38.5% # 3.730 min
dwb_blackbox_SA('RT', 'Google_NLP', 100)        # 26.5% # 1.993 min
dwb_blackbox_SA('RT', 'IBM_Watson', 100)        # 26.0% # 3.069 min
dwb_blackbox_SA('RT', 'Microsoft_Azure', 100)   # 38.5% # 1.739 min
dwb_blackbox_SA('RT', 'AWS_Comprehend', 100)    # 31.0% # 0.976 min





