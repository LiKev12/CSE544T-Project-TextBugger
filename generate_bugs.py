

import random
import pprint
import gensim.models.keyedvectors as word2vec
import numpy as np
from scipy import spatial
import pandas as pd 
import csv
import json






def generateBugs(word, glove_vectors):

    bugs = {"insert": word, "delete": word, "swap": word, "sub_C": word, "sub_W": word}

    if (len(word) <= 2):
        return bugs

    bugs["insert"] = bug_insert(word)
    bugs["delete"] = bug_delete(word)
    bugs["swap"] = bug_swap(word)
    bugs["sub_C"] = bug_sub_C(word)
    bugs["sub_W"] = bug_sub_W(word, glove_vectors)

    # pprint.pprint(bugs)

    return bugs

def bug_insert(word):
    if (len(word) >= 6):
        return word
    res = word
    point = random.randint(1, len(word)-1)
    res = res[0:point] + " " + res[point:]
    return res



def bug_delete(word):

    res = word
    point = random.randint(1, len(word)-2)
    res = res[0:point] + res[point+1:]
    # print("hi")
    # print(res[7:])
    return res


def bug_swap(word):
    if (len(word) <= 4):
        return word
    res = word
    points = random.sample(range(1, len(word)-1), 2)
    # print(points)
    a = points[0]
    b = points[1]

    res = list(res)
    w = res[a]
    res[a] = res[b]
    res[b] = w
    res = ''.join(res)
    return res

def bug_sub_C(word):
    res = word
    key_neighbors = get_key_neighbors()
    point = random.randint(0,len(word)-1)

    if word[point] not in key_neighbors:
        return word
    choices = key_neighbors[word[point]]
    subbed_choice = choices[random.randint(0,len(choices)-1)]
    res = list(res)
    res[point] = subbed_choice
    res = ''.join(res)

    return res

def bug_sub_W(word, glove_vectors):
    if word not in glove_vectors:
        return word

    closest_neighbors = find_closest_words(glove_vectors[word], glove_vectors)[1:6]
    
    return random.choice(closest_neighbors)
    # return closest_neighbors # Change later



def get_key_neighbors():
    # By keyboard proximity
    neighbors = {
        "q": "was","w": "qeasd","e": "wrsdf","r": "etdfg","t": "ryfgh","y": "tughj","u": "yihjk","i": "uojkl","o": "ipkl","p": "ol",    
        "a": "qwszx","s": "qweadzx","d": "wersfxc","f": "ertdgcv","g": "rtyfhvb","h": "tyugjbn","j": "yuihknm","k": "uiojlm","l": "opk",      
        "z": "asx","x": "sdzc","c": "dfxv","v": "fgcb","b": "ghvn","n": "hjbm","m": "jkn"
    }

    # By visual proximity
    neighbors['i'] += '1'
    neighbors['l'] += '1'
    neighbors['z'] += '2'
    neighbors['e'] += '3'
    neighbors['a'] += '4'
    neighbors['s'] += '5'
    neighbors['g'] += '6'
    neighbors['b'] += '8'
    neighbors['g'] += '9'
    neighbors['q'] += '9'
    neighbors['o'] += '0'

    return neighbors

def find_closest_words(point, glove_vectors):
    return sorted(glove_vectors.keys(), key=lambda word: spatial.distance.euclidean(glove_vectors[word], point))











