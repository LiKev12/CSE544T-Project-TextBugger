import sys
from collections import OrderedDict
import random
import pprint
from textbugger_utils import get_prediction_given_tokens, getSemanticSimilarity, transform_to_feature_vector, get_word_importances_for_whitebox, generateBugs
import numpy as np

class WhiteBox():

    def __init__(self, X, y, F, epsilon, model_type, glove_vectors, embed_map, dataset): 
        self.X = X # Sentence Tokens Ex. ['I','enjoy','playing','outside']
        self.y = y # Predicted score: 0/1
        self.F = F # Classifier/Model 
        self.epsilon = epsilon
        self.model_type = model_type
        self.glove_vectors = glove_vectors
        self.embed_map = embed_map
        self.dataset = dataset

    def whiteBoxAttack(self):
        # Lines 2-5: Compute importance C

        W_ordered = get_word_importances_for_whitebox(self.X, self.y, self.F, self.model_type, self.glove_vectors, self.embed_map, self.dataset)
        # print(W_ordered)

        # Lines 6-14: SelectBug and Iterate
        x_prime = self.X # Initialize x_prime = X
        num_words_total = len(W_ordered)
        num_perturbed = 0


        # print("Original (Score: {}): \n{}".format(self.y, x_prime))
        for x_i in W_ordered:
            bug = self.selectBug(x_i, x_prime)
            x_prime = self.replaceWithBestBug(x_prime, x_i, bug)
            prediction_proba = get_prediction_given_tokens(self.model_type, self.F, x_prime, glove_vectors = self.glove_vectors, embed_map = self.embed_map, dataset = self.dataset)
            prediction = np.round(prediction_proba,0)
            num_perturbed += 1

            # print("{} => {}".format(x_i, bug))
            # print(x_prime)
            # print("Score: {}".format(prediction_proba))


            if getSemanticSimilarity(self.X, x_prime, self.epsilon) <= self.epsilon:
                return None
            elif prediction != self.y:
                return x_prime,float(num_perturbed/num_words_total)
        # print("None found")
        return None

    def selectBug(self, original_word, x_prime):
        bugs = generateBugs(original_word, self.glove_vectors, typo_enabled = True)
        # bugs = generateBugs(original_word, self.glove_vectors)
        
        max_score = float('-inf')
        best_bug = original_word

        bug_tracker = {}
        for bug_type, b_k in bugs.items():
            candidate_k = self.getCandidate(original_word, b_k, x_prime)
            # print("ORIGINAL WORD: {} => {}".format(original_word, b_k))
            score_k = self.getScore(candidate_k, x_prime)
            if (score_k > max_score):
                best_bug = b_k # Update best bug
                max_score = score_k
            bug_tracker[b_k] = score_k
        # print(bug_tracker)
        
        return best_bug

    def getCandidate(self, original_word, new_bug, x_prime):
        tokens = x_prime
        new_tokens = [new_bug if x == original_word else x for x in tokens]
        return new_tokens


    def getScore(self, candidate, x_prime):

        x_prime_proba = get_prediction_given_tokens(self.model_type, self.F, x_prime, glove_vectors = self.glove_vectors, embed_map = self.embed_map, dataset = self.dataset)
        x_prime_with_bug_proba = get_prediction_given_tokens(self.model_type, self.F, candidate, glove_vectors = self.glove_vectors, embed_map = self.embed_map, dataset = self.dataset)
        # print('scoreX'.format(x_prime_proba))        
        # print('score_bug {}'.format(x_prime_with_bug_proba))

        if self.y == 1:
            score = x_prime_proba - x_prime_with_bug_proba
        else:
            score = x_prime_with_bug_proba - x_prime_proba
        return score

    def replaceWithBestBug(self, x_prime, x_i, bug):
        tokens = x_prime
        new_tokens = [bug if x == x_i else x for x in tokens]
        return new_tokens