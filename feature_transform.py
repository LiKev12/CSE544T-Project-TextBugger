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
import random


def transform_to_feature_vector(tokens, glove_vectors):
    vectors = []
    for token in tokens:
        if token in glove_vectors:
            vect = glove_vectors[token]
            vectors.append(vect)
        else:
            # sampling from the uniform distribution in [-0.1, 0.1]
            vect = [(random.random()/5)-0.1 for i in range(300)]
            vectors.append(vect)

    means = np.mean(vectors, axis=0)
    return [means] # [ [x11,x12,x13,...]  ]



