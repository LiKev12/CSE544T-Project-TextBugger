
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



def testWhiteBoxSentimentAnalysis(data_type, model_type):
    ## Import glove-vectors once
    glove_vectors = json.load( open( "glove_final.json", "rb") )
    embed_map = pickle.load( open( "datasets/embed_map.p", "rb" ) )
    embed_map = pickle.load( open( "datasets/embed_map.p", "rb" ) )

    ## Get Dataset (2 types: IMDB, RT)
    if (data_type == 'IMDB'):
        data = pickle.load( open( "datasets/IMDB/IMDB_tokens.p", "rb" ) )
    elif (data_type == 'RT'):
        data = pickle.load( open( "datasets/RT/RT_tokens.p", "rb" ) )

    ## Get Model (3 types: LR, LSTM, CNN)
    if (model_type == 'LR'):
        if (data_type == 'IMDB'):
            model = pickle.load( open( "models/LR/LR_SA_IMDB.p", "rb" ))
        elif(data_type == 'RT'):
            model = pickle.load( open( "models/LR/LR_SA_RT.p", "rb" ))

    elif (model_type == 'LSTM'):
        if (data_type == 'IMDB'):
            model = pickle.load( open( "models/LSTM/LSTM_SA_IMDB.p", "rb" ))
        elif(data_type == 'RT'):
            model = pickle.load( open( "models/LSTM/LSTM_SA_RT.p", "rb" ))

    elif (model_type == 'CNN'):
        if (data_type == 'IMDB'):
            model = pickle.load( open( "models/CNN/CNN_SA_IMDB.p", "rb" ))
        elif(data_type == 'RT'):
            model = pickle.load( open( "models/CNN/CNN_SA_RT.p", "rb" ))

    num_successes = 0 
    total_docs = 0

    for key1 in data:
        for key2 in data[key1]:
            docs = data[key1][key2]
            for doc in docs:
                y = get_prediction_given_tokens(model_type, model, doc, glove_vectors = glove_vectors, embed_map = embed_map, dataset=data_type)
                if (np.abs(y - 0.5) > 0.1):
                    # print("IMPOSSIBLE")
                    continue
                y = np.round(y,0)
                whitebox = WhiteBox(doc,y,model, 0.8, model_type, glove_vectors, embed_map, data_type)
                res = whitebox.whiteBoxAttack()
                if res != None:
                    num_successes += 1
                    percent_perturbed = res[1]
                    print("Successful adversary. Fraction of original input perturbed: {}".format(np.round(percent_perturbed,2)))
                total_docs += 1

    print("{} successful adversaries out of {} total documents. Success rate = {}".format(num_successes,total_docs,np.round(num_successes/total_docs,2)))







# testWhiteBoxSentimentAnalysis('IMDB','LSTM')  # LSTM - IMDB
testWhiteBoxSentimentAnalysis('RT','LSTM')    # LSTM - RT
# testWhiteBoxSentimentAnalysis('IMDB','CNN')   # CNN  - IMDB
# testWhiteBoxSentimentAnalysis('RT','CNN')     # CNN  - RT
# testWhiteBoxSentimentAnalysis('IMDB','LR')    # LR - RT
# testWhiteBoxSentimentAnalysis('RT','LR')      # LR - IMDB











