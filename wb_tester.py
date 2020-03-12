from white_box import WhiteBox

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






## Set X, find y
# doc = "It had been her dream for years but Dana had failed to take any action toward making it come true. There had always been a good excuse to delay or prioritize another project. As she woke, she realized she was once again at a crossroads. Would it be another excuse or would she finally find the courage to pursue her dream? Dana rose and took her first step."
# X = nltk.word_tokenize(doc)
# X_feat = transform_to_feature_vector(X, glove_vectors)



# y = model.predict(X_feat)[0]


def testWhiteBoxSentimentAnalysis():
    ## Import glove-vectors once
    with open('Glove_IMDB.json') as f:
        glove_vectors = json.load(f)

    ## Import model
    modelFileName = 'Sentiment_Analysis/White_Box/Models/LogisticRegression_IMDB.pkl'
    with open(modelFileName, 'rb') as fid:
        model = pickle.load(fid)
    with open('data/IMDB_text_tokenized.p','rb') as fp:
        data = pickle.load(fp)


    num_successes = 0 
    percent_perturbed = 0

    for key1 in data:
        for key2 in data[key1]:
            docs = data[key1][key2]
            for doc in docs:
                X_feat = transform_to_feature_vector(doc, glove_vectors)
                y = model.predict(X_feat)[0]
                whitebox = WhiteBox(doc,y,model,0.8,glove_vectors)
                res = whitebox.whiteBoxAttack()
                if res != None:
                    num_successes += 1
                    percent_perturbed += res[1]
                    print(num_successes)
                    print(percent_perturbed)
                    return



    print(num_successes)
    print(percent_perturbed)


# whitebox = WhiteBox(X,y,model,0.8, glove_vectors)
# whitebox.whiteBoxAttack()


testWhiteBoxSentimentAnalysis()