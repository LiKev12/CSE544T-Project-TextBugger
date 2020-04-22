
import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split
import nltk
import time
import pprint
from textbugger_utils import get_prediction_given_tokens
from whitebox import WhiteBox
import keras
import random


def testWhiteBox(data_type, model_type, num_samples):

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

    for doc in pos_samples:
        # print(sample_id)
        sample_id+=1
        y = get_prediction_given_tokens(model_type, model, doc, glove_vectors = glove_vectors, embed_map = embed_map, dataset=data_type)
        y = np.round(y,0)
        whitebox = WhiteBox(doc,y,model, 0.8, model_type, glove_vectors, embed_map, data_type)
        res = whitebox.whiteBoxAttack()
        if res != None:
            num_successes += 1
            percent_perturbed.append(res[1])
            # print("Successful adversary. Fraction of original input perturbed: {}".format(np.round(res[1],2)))

    for doc in neg_samples:
        # print(sample_id)
        sample_id+=1
        y = get_prediction_given_tokens(model_type, model, doc, glove_vectors = glove_vectors, embed_map = embed_map, dataset=data_type)
        y = np.round(y,0)
        whitebox = WhiteBox(doc,y,model, 0.8, model_type, glove_vectors, embed_map, data_type)
        res = whitebox.whiteBoxAttack()
        if res != None:
            num_successes += 1
            percent_perturbed.append(res[1])
            # print("Successful adversary. Fraction of original input perturbed: {}".format(np.round(res[1],2)))
    
    total_docs = 2 * num_samples
    success_rate = np.round((num_successes/total_docs)*100,3)
    perturb_rate = np.round(np.mean(percent_perturbed)*100,3)
    # print("{} successful adversaries out of {} total documents. Success rate = {}".format(num_successes,total_docs,success_rate))
    print("Avg % Perturbed: {}".format(perturb_rate))
    print("{} | {} | {}".format(data_type, model_type, success_rate))





## NORMAL (WITHOUT NOVEL)]
# testWhiteBox('IMDB','LR',10)      # LR - RT       4.87% | 100%
# testWhiteBox('RT','LR',10)        # LR - IMDB     17.7% | 95%
# testWhiteBox('IMDB','LSTM',10)    # LSTM - IMDB   45.3% | 34.1%
# testWhiteBox('RT','LSTM',10)      # LSTM - RT     56.5% | 70.3%
# testWhiteBox('IMDB','CNN',5)      # CNN  - IMDB   5.25% | 96%
# testWhiteBox('RT','CNN',5)        # CNN  - RT     41.5% | 98%


# testWhiteBox('Kaggle','LR',10)    # LR - Kaggle   31.2% | 90.8%
# testWhiteBox('Kaggle','LSTM',10)  # LSTM - Kaggle 16.3% | 23.5%
# testWhiteBox('Kaggle','CNN',10)   # CNN - Kaggle  26.8% | 80.3%

## NOVEL
# testWhiteBox('IMDB','LR',10)      # LR - RT       4.99% | 100%
# testWhiteBox('RT','LR',10)        # LR - IMDB     17.2% | 98%
# testWhiteBox('IMDB','LSTM',10)    # LSTM - IMDB   38.1% | 40%
# testWhiteBox('RT','LSTM',10)      # LSTM - RT     37.8% | 60%
# testWhiteBox('IMDB','CNN',5)      # CNN  - IMDB    4.1% | 100%
# testWhiteBox('RT','CNN',5)        # CNN  - RT     30.3% | 90%


# testWhiteBox('Kaggle','LR',10)    # LR - Kaggle   28.3% | 91.3%
# testWhiteBox('Kaggle','LSTM',10)  # LSTM - Kaggle 10.5% | 28.3%
# testWhiteBox('Kaggle','CNN',10)   # CNN - Kaggle  21.6% | 85.7%









