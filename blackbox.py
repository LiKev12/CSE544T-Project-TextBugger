
import sys
from collections import OrderedDict
import random
import pprint
import numpy as np
from textbugger_utils import get_prediction_given_tokens, getSemanticSimilarity, transform_to_feature_vector, get_word_importances_for_whitebox, generateBugs
from textbugger_utils import get_blackbox_classifier_score
import nltk
from spacy.lang.en import English # updated
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

class BlackBox():
    
    def __init__(self, X, y,epsilon, model_type, glove_vectors, dataset): 
        self.X = X # Sentence Tokens Ex. ['I','enjoy','playing','outside']
        self.y = y # Predicted score: 0/1
        self.epsilon = epsilon
        self.model_type = model_type
        self.glove_vectors = glove_vectors
        self.dataset = dataset

    def blackBoxAttack(self):
        sentences_of_document = self.get_sentences()
        ranked_sentences = self.rank_sentences(sentences_of_document)
        x_prime = self.X.copy()

        num_perturbed = 0
        num_words_total = len(self.X)

        for sentence_index in ranked_sentences:
            ranked_words = self.get_importances_of_words_in_sentence(sentences_of_document[sentence_index])
            for word in ranked_words:
                bug = self.selectBug(word, x_prime)
                x_prime = self.replaceWithBestBug(x_prime, word, bug)
                x_prime_sentence = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(x_prime)
                prediction_proba = get_blackbox_classifier_score(self.model_type, x_prime_sentence)
                prediction = np.round(prediction_proba,0)
                num_perturbed += 1

                print("{} => {}".format(word, bug))
                print("Score: {}".format(prediction_proba))

                if getSemanticSimilarity(self.X, x_prime, self.epsilon) <= self.epsilon:
                    return None
                elif prediction != self.y:
                    return x_prime,float(num_perturbed/num_words_total)
        print("None found")
        return None


    def get_sentences(self):
        original_review = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(self.X)
        nlp = English()
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        doc = nlp(original_review)
        sentences = [sent.string.strip() for sent in doc.sents]
        return sentences
            
    def rank_sentences(self, sentences):
        map_sentence_to_importance = {}
        for i in range(len(sentences)):
            classifier_score = get_blackbox_classifier_score(self.model_type, sentences[i])
            if self.y == 0:
                importance = 0.5 - classifier_score
            else:
                importance = classifier_score - 0.5

            if (importance > 0):
                map_sentence_to_importance[i] = importance

        sentences_sorted_by_importance = {k: v for k, v in sorted(map_sentence_to_importance.items(), key=lambda item: -item[1])}

        return sentences_sorted_by_importance



    def get_importances_of_words_in_sentence(self, sentence):
        sentence_tokens = nltk.word_tokenize(sentence)
        word_importances = {}
        for curr_token in sentence_tokens:
            sentence_tokens_without = [token for token in sentence_tokens if token != curr_token]
            sentence_without = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(sentence_tokens_without)
            word_score =  get_blackbox_classifier_score(self.model_type, sentence_without)
            if (self.y == 1):
                word_importance = 1 - word_score
            else:
                word_importance = word_score

            word_importances[curr_token] = word_importance
        word_importances = {k: v for k, v in sorted(word_importances.items(), key=lambda item: -item[1])}
        print(word_importances)

        return word_importances





    def selectBug(self, original_word, x_prime):
        bugs = generateBugs(original_word, self.glove_vectors)
    
        max_score = float('-inf')
        best_bug = original_word

        bug_tracker = {}
        for bug_type, b_k in bugs.items():
            candidate_k = self.getCandidate(original_word, b_k, x_prime)
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
        x_prime_sentence = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(x_prime)
        x_prime_proba = get_blackbox_classifier_score(self.model_type, x_prime_sentence)
        candidate_sentence = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(candidate)
        candidate_proba = get_blackbox_classifier_score(self.model_type, candidate_sentence)

        if self.y == 1:
            score = x_prime_proba - candidate_proba
        else:
            score = candidate_proba - x_prime_proba
        return score

    def replaceWithBestBug(self, x_prime, x_i, bug):
        tokens = x_prime
        new_tokens = [bug if x == x_i else x for x in tokens]
        return new_tokens