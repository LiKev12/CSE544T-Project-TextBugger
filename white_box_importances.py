from operator import add
import numpy as np
import json
import _pickle as pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
import smart_open
import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
import nltk
import time
import pprint
import pandas as pd

from feature_transform import transform_to_feature_vector


def get_word_importances_for_whitebox(tokens, glove_vectors):
    # start = time.time()
    # print("LOADING...")
    # with open('Glove_RT.json') as f:
    #     glove_vectors = json.load(f)
    # end = time.time()
    # print("DONE LOADING")
    # print(str((end-start)/60) + " minutes")



    ## Transform tokens to the avg feature vector
    vector = transform_to_feature_vector(tokens, glove_vectors)


    ## Load model, get prediction for whole document
    modelFileName = 'Sentiment_Analysis/White_Box/Models/LogisticRegression_RT.pkl'
    with open(modelFileName, 'rb') as fid:
        model = pickle.load(fid)

    pred = model.predict(vector)[0]
    # print("Prediction: " + str(pred))

    pred_proba = model.predict_proba(vector)[0][1]


    ## Compute importance for each word
    excludes = get_excludes(tokens)         # To see the relative importance of each word, remove that word and predict
    

    
    JM = {}
    for ex_word, ex_tokens in excludes.items():
        # print(ex_tokens)
        ex_vect = transform_to_feature_vector(ex_tokens, glove_vectors)

        ex_pred_proba = model.predict_proba(ex_vect)[0][1]

        if (pred == 1):
            C = pred_proba - ex_pred_proba
        else:
            C = ex_pred_proba - pred_proba

        JM[ex_word] = C

    # print(JM)
    ordered_list_by_importance = getImportances(JM)
    # print(ordered_list_by_importance)

    return ordered_list_by_importance





def get_excludes(l1):

    res = {}
    for el in l1:
        sub = [x for x in l1 if x != el]
        res[el] = sub
    return res


def getImportances(JM):
    df = pd.DataFrame(JM.items(), columns=['Word', 'C'])

    df = df.sort_values('C', ascending=False)
    return df['Word'].tolist()



## Testing

with open('Glove_RT.json') as f:
    glove_vectors = json.load(f)

get_word_importances_for_whitebox("love laugh perfect sad", glove_vectors) 
# get_word_importances_for_whitebox("sad sad hate I worst.")
