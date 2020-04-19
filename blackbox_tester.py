
import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split
import nltk
import time
import pprint
import keras
import random
from blackbox import BlackBox
from textbugger_utils import get_blackbox_classifier_score
from nltk.tokenize.treebank import TreebankWordDetokenizer


## Classifers used
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


def testBlackBoxSentimentAnalysis(data_type, model_type):
    ## Import glove-vectors once
    glove_vectors = json.load( open( "glove_final.json", "rb") )
    embed_map = pickle.load( open( "datasets/embed_map.p", "rb" ) )

    ## Get Dataset (2 types: IMDB, RT)
    if (data_type == 'IMDB'):
        data = pickle.load( open( "datasets/IMDB/IMDB_tokens.p", "rb" ) )
    elif (data_type == 'RT'):
        data = pickle.load( open( "datasets/RT/RT_tokens.p", "rb" ) )


    num_successes = 0 
    total_docs = 0



    for key1 in data:
        for key2 in data[key1]:
            token_lists = data[key1][key2]
            for token_list in token_lists:
                sentence = TreebankWordDetokenizer().detokenize(token_list)
                y = get_blackbox_classifier_score(model_type, sentence)
                y_class = np.round(y,0)
                print('Original Score: {} | Label: {}'.format(y,y_class))

                blackbox = BlackBox(token_list,y_class,0.8, model_type, glove_vectors, data_type)
                res = blackbox.blackBoxAttack()
                if res != None:
                    num_successes += 1
                    percent_perturbed = res[1]
                    print("Successful adversary. Fraction of original input perturbed: {}".format(np.round(percent_perturbed,2)))
                total_docs += 1

    print("{} successful adversaries out of {} total documents. Success rate = {}".format(num_successes,total_docs,np.round(num_successes/total_docs,2)))



model_type = 'Google_NLP'
# testBlackBoxSentimentAnalysis('IMDB', model_type)
testBlackBoxSentimentAnalysis('RT', model_type)

