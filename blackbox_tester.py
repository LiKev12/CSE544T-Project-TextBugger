
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


def testBlackBoxSentimentAnalysis(data_type, model_type, num_samples):
    start = time.time()
    ## Import glove-vectors once
    if (data_type == 'RT' or data_type == 'IMDB'):
        glove_vectors = json.load( open( "glove_final.json".format(data_type, data_type), "rb") )
    elif (data_type == 'Kaggle'):
        glove_vectors = json.load( open( "datasets/{}/glove_kaggle.json".format(data_type, data_type), "rb") )

    
    embed_map = pickle.load( open( "datasets/{}/{}_embed_map.p".format(data_type, data_type), "rb" ) )

    ## Get Dataset (3 types: IMDB, RT, Kaggle)
    if (data_type == 'IMDB'):
        data = pickle.load( open( "datasets/IMDB/IMDB_tokens.p", "rb" ) )
    elif (data_type == 'RT'):
        data = pickle.load( open( "datasets/RT/RT_tokens.p", "rb" ) )
    elif(data_type == 'Kaggle'):
        data = pickle.load(open("datasets/Kaggle/Kaggle_tokens.p","rb"))

    ## Get Model (3 types: LR, LSTM, CNN)
    if (model_type == 'LR'):
        if (data_type == 'IMDB'):
            model = pickle.load( open( "models/LR/LR_SA_IMDB.p", "rb" ))
        elif(data_type == 'RT'):
            model = pickle.load( open( "models/LR/LR_SA_RT.p", "rb" ))
        elif(data_type == 'Kaggle'):
            model = pickle.load( open( "models/LR/LR_TCD_Kaggle.p", "rb" ))
    elif (model_type == 'LSTM'):
        if (data_type == 'IMDB'):
            model = pickle.load( open( "models/LSTM/LSTM_SA_IMDB.p", "rb" ))
        elif(data_type == 'RT'):
            model = pickle.load( open( "models/LSTM/LSTM_SA_RT.p", "rb" ))
        elif(data_type == 'Kaggle'):
            model = pickle.load( open( "models/LSTM/LSTM_TCD_Kaggle.p", "rb" ))
    elif (model_type == 'CNN'):
        if (data_type == 'IMDB'):
            model = pickle.load( open( "models/CNN/CNN_SA_IMDB.p", "rb" ))
        elif(data_type == 'RT'):
            model = pickle.load( open( "models/CNN/CNN_SA_RT.p", "rb" ))
        elif(data_type == 'Kaggle'):
            model = pickle.load( open( "models/CNN/CNN_TCD_Kaggle.p", "rb" ))

    #---- DONE LOADING ----------
    end = time.time()
    print("DONE LOADING: {} minutes".format(np.round((end-start)/60),4))

    num_successes = 0 
    sample_id = 1
    percent_perturbed = []

    pos_samples = data['test']['pos'][0:num_samples]
    neg_samples = data['test']['neg'][0:num_samples]


    num_successes = 0 
    total_docs = 0



    for token_list in pos_samples:
        sentence = TreebankWordDetokenizer().detokenize(token_list)
        y = get_blackbox_classifier_score(model_type, sentence)
        y_class = np.round(y,0)
        # print(sentence)
        # print('Original Score: {} | Label: {}'.format(y,y_class))

        blackbox = BlackBox(token_list,y_class,0.8, model_type, glove_vectors, data_type)
        res = blackbox.blackBoxAttack()
        if res != None:
            num_successes += 1
            percent_perturbed.append(res[1])
        total_docs += 1

    for token_list in neg_samples:
        sentence = TreebankWordDetokenizer().detokenize(token_list)
        y = get_blackbox_classifier_score(model_type, sentence)
        y_class = np.round(y,0)
        # print(sentence)
        # print('Original Score: {} | Label: {}'.format(y,y_class))

        blackbox = BlackBox(token_list,y_class,0.8, model_type, glove_vectors, data_type)
        res = blackbox.blackBoxAttack()
        if res != None:
            num_successes += 1
            percent_perturbed.append(res[1])
            # print("Successful adversary. Fraction of original input perturbed: {}".format(np.round(percent_perturbed,2)))
        total_docs += 1
    

    total_docs = 2 * num_samples
    success_rate = np.round((num_successes/total_docs)*100,3)
    perturb_rate = np.round(np.mean(percent_perturbed)*100,3)
    print("Avg % Perturbed: {}".format(perturb_rate))
    print("{} | {} | {}".format(data_type, model_type, success_rate))



# testBlackBoxSentimentAnalysis('RT', 'Google_NLP')
# testBlackBoxSentimentAnalysis('RT', 'IBM_Watson')
# testBlackBoxSentimentAnalysis('RT', 'Microsoft_Azure')
# testBlackBoxSentimentAnalysis('RT', 'AWS_Comprehend')
# testBlackBoxSentimentAnalysis('RT', 'FB_fastText')

# testBlackBoxSentimentAnalysis('IMDB', 'Google_NLP')
# testBlackBoxSentimentAnalysis('IMDB', 'IBM_Watson')
# testBlackBoxSentimentAnalysis('IMDB', 'Microsoft_Azure')
# testBlackBoxSentimentAnalysis('IMDB', 'AWS_Comprehend')
# testBlackBoxSentimentAnalysis('IMDB', 'FB_fastText')

# testBlackBoxSentimentAnalysis('Kaggle', 'Google_NLP', 2)
# testBlackBoxSentimentAnalysis('Kaggle', 'IBM_Watson', 2)
# testBlackBoxSentimentAnalysis('Kaggle', 'Microsoft_Azure',10)
# testBlackBoxSentimentAnalysis('Kaggle', 'AWS_Comprehend',10)
# testBlackBoxSentimentAnalysis('Kaggle', 'FB_fastText',2) 16.2%P | 75%A