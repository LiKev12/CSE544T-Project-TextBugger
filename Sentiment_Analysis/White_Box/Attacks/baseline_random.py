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




def baseline_random(input_text):

    input_text = preprocess(input_text)
    lines = input_text.split(".")
    numWords = 0
    perturbed = 0
    for i in range(0,len(lines)):

        output = randomlyModify(lines[i])
        lines[i] = output[0]
        numWords += output[1]
        perturbed += output[2]
        
    print(perturbed)    
    print(numWords)
    print(round(float(perturbed/numWords),3))

    res = ".".join(lines)
    print(res)
    return res


def randomlyModify(point):

    words = point.split(" ")
    numWords = len(words)
    perturbed = 0

    for i in range(0,len(words)):
        rn = random.random()
        if (rn> 0.90):
            perturbed += 1
            words[i] = getModifiedWord(words[i])

    return " ".join(words), numWords, perturbed

def getModifiedWord(word):
    point = word
    return word
    


sentence = "If you can imagine a furry humanoid seven feet tall, with the face of an intelligent gorilla and the braincase of a man, you'll have a rough idea of what they looked like -- except for their teeth. The canines would have fitted better in the face of a tiger, and showed at the corners of their wide, thin-lipped mouths, giving them an expression of ferocity."
baseline_random(sentence)