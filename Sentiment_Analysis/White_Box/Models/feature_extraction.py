import csv
import pprint
import sys
import pandas as pd
import os
import re
import numpy as np
import json

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score
import pickle

from operator import add

try:
    import ujson as json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        import json





def featureExtraction_IMDB_Train(self, data):
    with open('../../../glove_vectors.json') as json_file:
        glove_vectors = json.load(json_file)
    print("DONE")

    for key1 in data:
        for key2 in data[key1]:
            feature_vectors = []
            for point in data[key1][key2]:
                print(len(point))
                vector = self.getFeatureVector(point, glove_vectors)
                feature_vectors.append(vector) # feature_vectors contains the mean of each sentence/input

            data[key1][key2] = feature_vectors


    # Do one time for future use
    with open('../../../data/IMDB_vectors.p', 'wb') as fp:
        pickle.dump(data, fp)

def featureExtraction_RT(self, data):
    print("Extracting RT...")
    with open('../../../glove_vectors.json') as json_file:
        glove_vectors = json.load(json_file)
    print("DONE LOADING")

    for key1 in data:
        feature_vectors = []
        for point in data[key1]:
            print(len(point))
            vector = self.getFeatureVector(point, glove_vectors)
            feature_vectors.append(vector)

        data[key1] = feature_vectors


    ## Do one time for future use
    with open('../../../data/RT_vectors.p', 'wb') as fp:
        pickle.dump(data, fp)


def getFeatureVector(self, point, glove_vectors):
    numTokens = 0
    vect = np.asarray([0] * 300)
    for token in point:
        if token in glove_vectors:
            vect = np.add(vect, np.asarray(glove_vectors[token]))
            numTokens += 1
    vect = np.divide(vect, numTokens)
    return vect

