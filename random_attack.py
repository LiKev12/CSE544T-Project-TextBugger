import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split
import nltk
import time
import pprint
from textbugger_utils import get_prediction_given_tokens, generateBugs
from whitebox import WhiteBox
import keras
import random


def test_random_attack(data_type, model_type, num_samples):
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


    pos_samples = data['test']['pos'][0:num_samples]
    neg_samples = data['test']['neg'][0:num_samples]

    num_success = 0

    for sample in pos_samples:
        ori_doc = sample
        before_pred = get_prediction_given_tokens(model_type, model, ori_doc, glove_vectors = glove_vectors, embed_map = embed_map, dataset = data_type)

        adv_doc = get_adv_random(ori_doc,50, glove_vectors)
        after_pred = get_prediction_given_tokens(model_type, model, adv_doc, glove_vectors = glove_vectors, embed_map = embed_map, dataset = data_type) 

        before_class = np.round(before_pred,0)
        after_class = np.round(after_pred,0)

        if (before_class != after_class):
            num_success += 1

    total = 2 * num_samples

    print("{} | {} | {}".format(data_type, model_type, np.round((num_success/total)*100,3)))

def get_adv_random(ori_doc, perturbed_percent, glove_vectors):
    doc = ori_doc.copy()
    for i in range(len(doc)):
        token = doc[i]
        rn = random.random() * 100
        if (rn < perturbed_percent):
            bugs = generateBugs(token, glove_vectors, sub_w_enabled=False)
            chosen_bug = random.choice(list(bugs.items()))[1]
            doc[i] = chosen_bug
    return doc





## SA
# test_random_attack('IMDB','LR',10) # 2.8%
# test_random_attack('IMDB','LSTM',10) # 3.5%
# test_random_attack('IMDB','CNN',10) # 3.3%

# test_random_attack('RT','LR',10) # 25.4%
# test_random_attack('RT','LSTM',10) # 15.2%
# test_random_attack('RT','CNN',10) # 10.6%

## TCD
# test_random_attack('Kaggle','LR',10) # 10.3
# test_random_attack('Kaggle','LSTM',10) # 2.3%
# test_random_attack('Kaggle','CNN',10) # 13.2%