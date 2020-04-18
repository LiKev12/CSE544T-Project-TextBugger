import sys
from collections import OrderedDict
import random
import pprint
from whitebox_utils import get_prediction_given_tokens, getSemanticSimilarity, transform_to_feature_vector, get_word_importances_for_whitebox, generateBugs
import numpy as np

class WhiteBox():

    def __init__(self, X, y, F, epsilon, model_type, glove_vectors, embed_map): 
        self.X = X # Sentence Tokens Ex. ['I','enjoy','playing','outside']
        self.y = y # Predicted score: 0/1
        self.F = F # Classifier/Model 
        self.epsilon = epsilon
        self.model_type = model_type
        self.glove_vectors = glove_vectors
        self.embed_map = embed_map

    def whiteBoxAttack(self):
        print("whiteBoxAttack")
        # Lines 2-5: Compute importance C

        W_ordered = get_word_importances_for_whitebox(self.X, self.glove_vectors)
        print(W_ordered)

        # Lines 6-14: SelectBug and Iterate
        x_prime = self.X # Initialize x_prime = X
        num_words_total = len(W_ordered)
        num_perturbed = 0

        for x_i in W_ordered:
            bug = self.selectBug(x_i)
            x_prime = self.replaceWithBestBug(x_prime, x_i, bug)
            # prediction = self.F.predict(transform_to_feature_vector(x_prime, self.glove_vectors))[0]
            prediction_proba = get_prediction_given_tokens(self.model_type, self.F, x_prime, glove_vectors = self.glove_vectors, embed_map = self.embed_map)
            prediction = np.round(prediction_proba,0)
            num_perturbed += 1
            

            if getSemanticSimilarity(self.X, x_prime, self.epsilon) <= self.epsilon:
                return None
            elif prediction != self.y:
                return x_prime,float(num_perturbed/num_words_total)
        print("None found")
        return None

    def selectBug(self, original_word):
        bugs = generateBugs(original_word, self.glove_vectors)
        
        max_score = float('-inf')
        best_bug = original_word
        for bug_type, b_k in bugs.items():
            candidate_k = self.getCandidate(original_word, b_k)
            score_k = self.getScore(candidate_k)
            if (score_k > max_score):
                best_bug = b_k # Update best bug
                max_score = score_k
        
        # print(original_word + " => " + best_bug)
        return best_bug

    def getCandidate(self, original_word, new_bug):
        tokens = self.X
        new_tokens = [new_bug if x == original_word else x for x in tokens]
        return new_tokens


    def getScore(self, candidate):
        X_feat = transform_to_feature_vector(self.X, self.glove_vectors)
        
        y_proba = self.F.predict_proba(X_feat)[0][1]
        y_pred = self.y

        C_feat = transform_to_feature_vector(candidate, self.glove_vectors)
        candidate_proba = self.F.predict_proba(C_feat)[0][1]

        if y_pred == 1:
            score = y_proba - candidate_proba
        else:
            score = candidate_proba - y_proba
        return score

    def replaceWithBestBug(self, x_prime, x_i, bug):
        tokens = x_prime
        new_tokens = [bug if x == x_i else x for x in tokens]
        return new_tokens