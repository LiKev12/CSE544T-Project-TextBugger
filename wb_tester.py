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



## Import glove-vectors once
with open('Glove_RT.json') as f:
    glove_vectors = json.load(f)

## Import model
modelFileName = 'Sentiment_Analysis/White_Box/Models/LogisticRegression_RT.pkl'
with open(modelFileName, 'rb') as fid:
    model = pickle.load(fid)


## Set X, find y
doc = "love laugh perfect sad"
X = nltk.word_tokenize(doc)
X_feat = transform_to_feature_vector(X, glove_vectors)

y = model.predict(X_feat)[0]

whitebox = WhiteBox(X,y,model,0.2, glove_vectors)
whitebox.whiteBoxAttack()