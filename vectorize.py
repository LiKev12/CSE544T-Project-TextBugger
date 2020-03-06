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


def getTokens(sentence):
    tokens = nltk.word_tokenize(sentence)
    print(tokens)
    return tokens


def getFeatureVector(sentence, dataset):
    
    tokens = getTokens(sentence)
    if dataset=="RT":
        dictionary  = pickle.load("data/RT_vectors.p")


    numTokens = 0
    vect = np.asarray([0] * 300)
    for token in point:
        if token in glove_vectors:
            vect = np.add(vect, np.asarray(glove_vectors[token]))
            numTokens += 1
    vect = np.divide(vect, numTokens)
    return vect


sentence = '''My yardstick for measuring a movie's watch-ability is if I get squirmy. If I start shifting positions and noticing my butt is sore, the film is too long. This movie did not even come close to being boring. Predictable in some parts sure, but never boring.<br /><br />All of the other military branches have had love notes written about them and seen their recruitment levels go up, why not the Coast Guard too? They are definitely under-appreciated, until the day your boat sinks that is.<br /><br />The movie was very enjoyable and fun. Kevin Costner is perfect as the aging macho man who doesn't know when to quit. However, I was most impressed by Ashton Kutcher's performance. I have never liked him, never watched any of his TV shows and always considered him an immature ... well, punk. In this film, he does a great job! He is well on his way to having leading-man status. I think the film we were shown must have been an advance rough cut or something, because about 2/3 of the way in, the film stock turned very grainy, the sound level dropped and microphones were seen dropping down all over the place. Also at the viewing were representatives from the movie, looking for audience feedback - particularly on the parts of the film we didn't like.<br /><br />*****POSSIBLE SPOILER: The feedback I gave concerned a a couple of lines in the beginning. Kevin Costner comes home to see his wife, Sela Ward, packing her stuff up and moving out. He says, "Maybe I should be the one to move out." And she replies, "No, you don't know where anything is in this house; I should be the one to go." This doesn't make sense: If she knows the layout so well, Costner is right, he *should* be the one to leave.'''
getFeatureVector(sentence, '')





