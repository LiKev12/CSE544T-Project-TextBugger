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


class Logistic_Regression():  

    def featureExtraction(self, data):
        with open('../../../glove_vectors.json') as json_file:
            glove_vectors = json.load(json_file)
        print("DONE")

        for key1 in data:
            for key2 in data[key1]:
                feature_vectors = []
                for point in data[key1][key2]:
                    print(len(point))
                    vector = self.getFeatureVector(point, glove_vectors)
                    feature_vectors.append(vector)

                data[key1][key2] = feature_vectors


        ## Do one time for future use
        # with open('../../../data/IMDB_vectors.p', 'wb') as fp:
        #     pickle.dump(data, fp)


    def getFeatureVector(self, point, glove_vectors):
        numTokens = 0
        vect = np.asarray([0] * 300)
        for token in point:
            if token in glove_vectors:
                vect = np.add(vect, np.asarray(glove_vectors[token]))
                numTokens += 1
        vect = np.divide(vect, numTokens)
        return vect



    def load_data(self):


        ## Only do first time to get vectors
        # with open('../../../data/IMDB_text_tokenized.p', 'rb') as f:
        #     data = pickle.load(f)
        # data = self.featureExtraction(data)



        with open('../../../data/IMDB_vectors.p', 'rb') as f:
            data = pickle.load(f)

        X=[]
        y=[]

        # positive (12500) 1 
        X_pos = data["train"]["pos"]
        X.extend(X_pos)
        y.extend([1] * len(X_pos))

        # negative (12500) 0
        X_neg = data["train"]["neg"]
        X.extend(X_neg)
        y.extend([0] * len(X_neg))


        X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2)


        # Scale according to training data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        print(len(X_train))
        print(len(X_val))
        return X_train, X_val, y_train, y_val
    


    def train(self, X_train, y_train):
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # save the classifier
        with open('LogisticRegression.pkl', 'wb') as fid:
            pickle.dump(model, fid)    

        return model

    def test(self, model, X_val, y_val):
        y_pred = model.predict(X_val)
        print(accuracy_score(y_val, y_pred))
        return accuracy_score(y_val, y_pred)

    
if __name__ == "__main__":
    logreg = Logistic_Regression()
    # X_train, X_val, y_train, y_val = logreg.load_data()
    X_train, X_val, y_train, y_val = logreg.load_data_RT()
    model = logreg.train(X_train, y_train)
    logreg.test(model, X_val, y_val)



