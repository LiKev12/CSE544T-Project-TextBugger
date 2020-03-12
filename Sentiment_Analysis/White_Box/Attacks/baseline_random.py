import random
import pprint
import gensim.models.keyedvectors as word2vec
from baseline_preprocess import preprocess
import numpy as np
from scipy import spatial
import pandas as pd 
import csv
import json
import re

import sys
sys.path.insert(0, '../../../')

from generate_bugs import generateBugs
from feature_transform import transform_to_feature_vector
from semantic_similarity import getSemanticSimilarity

# from "../../../generate_bugs.py" import generateBugs
# from "../../../feature_transform.py" import transform_to_feature_vector

class Attack_Random():

    def __init__(self, X, F, epsilon, glove_vectors): 
        self.X = X # Sentence tokens ['I', 'enjoy','playing','outside']
        self.F = F
        self.epsilon = epsilon
        self.glove_vectors = glove_vectors


    def attack_random(self):
        
        y_pred_before = self.F.predict(transform_to_feature_vector(self.X, self.glove_vectors))
        
        X_prime = self.get_random()
        # print(self.X)
        # print(X_prime)
        y_pred_after = self.F.predict(transform_to_feature_vector(X_prime, self.glove_vectors))

        # print(self.X)
        # print(X_prime)

        ## Success = 1

        if getSemanticSimilarity(self.X, X_prime, self.epsilon) < self.epsilon:
            return 0

        if y_pred_after != y_pred_before:
            return 1
        else:
            return 0


    def get_random(self):
        
        X_prime = self.X.copy()
        for i in range(0,len(X_prime)):
            rn = random.random()
            if rn > 0.10:
                bugs = generateBugs(X_prime[i], self.glove_vectors)
                X_prime[i] = random.choice(list(bugs.values()))
            
        return X_prime
