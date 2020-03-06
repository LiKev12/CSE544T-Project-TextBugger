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

    def load_data_IMDB(self):
        ## Only do first time to get vectors
        # with open('../../../data/IMDB_text_tokenized.p', 'rb') as f:
        #     data = pickle.load(f)
        # data = self.featureExtraction_IMDB(data)



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



        ## Testing
        X_test=[]
        y_test=[]

        X_test.extend(data["test"]["pos"])
        y_test.extend([1] * len(data["test"]["pos"]))

        X_test.extend(data["test"]["neg"])
        y_test.extend([0] * len(data["test"]["neg"]))


        X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2)


        # Scale according to training data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        print(len(X_test))
        print(len(y_test))
        return X_train, X_val, y_train, y_val, X_test, y_test

    def load_data_RT(self):
    

        ## Only do first time to get vectors
        # with open('../../../data/RT_text_tokenized.p', 'rb') as f:
        #     data = pickle.load(f)
        # data = self.featureExtraction_RT(data)

        with open('../../../data/RT_vectors.p', 'rb') as f:
            data = pickle.load(f)

        X=[]
        y=[]

        # positive (12500) 1 
        X_pos = data["pos"]
        X.extend(X_pos)
        y.extend([1] * len(X_pos))

        # negative (12500) 0
        X_neg = data["neg"]
        X.extend(X_neg)
        y.extend([0] * len(X_neg))


        X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2)


        # Scale according to training data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        # print(len(X_train))
        # print(len(X_val))




        return X_train, X_val, y_train, y_val
    


    def train_IMDB(self, X_train, y_train):
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # save the classifier
        with open('LogisticRegression_IMDB.pkl', 'wb') as fid:
            pickle.dump(model, fid)    

        return model

    def train_RT(self, X_train, y_train):
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # save the classifier
        with open('LogisticRegression_RT.pkl', 'wb') as fid:
            pickle.dump(model, fid)    

        return model

    def validate(self, model, X_val, y_val):
        y_pred = model.predict(X_val)
        print("VALIDATION SCORE: " + str(accuracy_score(y_val, y_pred)))
        return accuracy_score(y_val, y_pred)

    def test_IMDB(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        print("TESTING SCORE: " + str(accuracy_score(y_test, y_pred)))
        return accuracy_score(y_test, y_pred)

    
if __name__ == "__main__":


    if len(sys.argv)<2:
        print("python3 model_LogisticRegression.py IMDB")
        sys.exit(1)

    dataset_used = sys.argv[1]

    if dataset_used == "IMDB":
        logreg = Logistic_Regression()
        X_train, X_val, y_train, y_val, X_test, y_test = logreg.load_data_IMDB()
        model = logreg.train_IMDB(X_train, y_train)
        logreg.validate(model, X_val, y_val)
        logreg.test_IMDB(model, X_test, y_test)
    elif dataset_used == "RT":
        logreg = Logistic_Regression()
        X_train, X_val, y_train, y_val = logreg.load_data_RT()
        model = logreg.train_RT(X_train, y_train)
        logreg.validate(model, X_val, y_val)
    else:
        print("Incorrect parameters: Enter either [IMDB] or [RT]")


