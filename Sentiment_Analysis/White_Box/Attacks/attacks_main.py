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

from baseline_random import Attack_Random




def test_attack_random(X, F, epsilon, glove_vectors):
    attack = Attack_Random(X, F, epsilon, glove_vectors)
    score = attack.attack_random()
    return score





## 1) Attack: Random

def call_text_attack_random():

    epsilon = 0.8



    ## RT / LR

    glove_vectors = util_get_glove_vectors('../../../Glove_RT.json')
    F = util_get_model('../Models/LogisticRegression_RT.pkl')

    with open('../../../data/RT_tokenized_TEST.p','rb') as fp:
        data = pickle.load(fp)

    Xs = data['pos']
    Xs = np.concatenate((data['pos'], data['neg']), axis=0)


    total_score = 0
    idx = 0
    for X in Xs:
        score = test_attack_random(X, F, epsilon, glove_vectors)
        total_score += score
        idx += 1
        print(idx)
        if idx > 500:
            break

    print(str(round(total_score* 100 / len(Xs),2)) + "%")




def util_get_glove_vectors(path_name):
    with open(path_name) as f:
        glove_vectors = json.load(f)
    return glove_vectors

def util_get_model(path_name):
    with open(path_name, 'rb') as fid:
        F = pickle.load(fid)
    return F








if __name__ == "__main__":
    call_text_attack_random()
