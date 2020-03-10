
from operator import add
import numpy as np

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
import time
import pprint
import pandas as pd

from generate_bugs import generateBugs
from sklearn.model_selection import train_test_split



def get_glove_IMDB():
    print("LOADING...")
    start = time.time()
    with open('glove_vectors.json') as f:
        glove_vectors = json.load(f)

    print("DONE LOADING GLOVE")
    end=time.time()
    print(str(end-start) + " seconds")
    print(str((end-start)/60) + " minutes")


    ## IMDB
    imdb_path = "data/IMDB_text_tokenized.p"

    # with open(imdb_path, 'rb') as f:
    #     imdb_tokens = pickle.load(f)

    # imdb_glove = {}
    # for key1 in imdb_tokens:
    #     for key2 in imdb_tokens[key1]:
    #         sentences = imdb_tokens[key1][key2]
    #         for sentence in sentences:
    #             for word in sentence:
    #                 if word in glove_vectors:
    #                     imdb_glove[word] = glove_vectors[word]

    # with open('Glove_IMDB.json', 'w') as f:
    #     json.dump(imdb_glove, f)


    ## Rotten Tomatoes
    rt_path = "data/RT_text_tokenized.p"

    with open(rt_path, 'rb') as f:
        rt_tokens = pickle.load(f)

    rt_glove = {}
    for key1 in rt_tokens:
        sentences = rt_tokens[key1]
        for sentence in sentences:
            for word in sentence:
                if word in glove_vectors:
                    rt_glove[word] = glove_vectors[word]

    with open('Glove_RT.json', 'w') as f:
        json.dump(rt_glove, f)





# get_glove_IMDB()









def testJM(sentence):

    tokens = nltk.word_tokenize(sentence)

    start = time.time()
    print("LOADING...")
    with open('Glove_RT.json') as f:
        glove_vectors = json.load(f)
    end = time.time()
    print("DONE LOADING")
    print(str((end-start)/60) + " minutes")



    ## Transform tokens to the avg feature vector
    vector = transform_to_feature_vector(tokens, glove_vectors)


    ## Load model, get prediction for whole document
    modelFileName = 'Sentiment_Analysis/White_Box/Models/LogisticRegression_RT.pkl'
    with open(modelFileName, 'rb') as fid:
        model = pickle.load(fid)

    pred = model.predict_proba(vector)[0][1]


    ## Compute importance for each word
    excludes = get_excludes(tokens)         # To see the relative importance of each word, remove that word and predict
    

    
    JM = {}
    for ex_word, ex_tokens in excludes.items():
        ex_vect = transform_to_feature_vector(ex_tokens, glove_vectors)

        ex_pred = model.predict_proba(ex_vect)[0][1]

        if (pred == 1):
            C = pred - ex_pred
        else:
            C = ex_pred - pred

        JM[ex_word] = C

    print(JM)
    ordered_list_by_importance = getImportances(JM)
    print(ordered_list_by_importance)

    return pred


def transform_to_feature_vector(tokens, glove_vectors):
    vectors = []
    for token in tokens:
        if token in glove_vectors:
            vect = glove_vectors[token]
            vectors.append(vect)

    means = np.mean(vectors, axis=0)
    return [means]



def get_excludes(l1):

    res = {}
    for el in l1:
        sub = [x for x in l1 if x != el]
        res[el] = sub
    return res


def getImportances(JM):
    df = pd.DataFrame(JM.items(), columns=['Word', 'C'])

    df = df.sort_values('C', ascending=False)
    return df['Word'].tolist()







# testJM("I like to eat all pies all the time.") # Positive
# testJM("I hate people when they are disgusting.") # Negative



# d1 = {"Hi": 7, "Joly": 12, "Ayy": 3, "JK": -100}
# df = pd.DataFrame(d1.items(), columns=['Word', 'C'])

# df = df.sort_values('C', ascending=False)
# print(df['Word'].tolist())


def test3():
    ls = ["hey", "hi", "ayy", "hi", "food"]
    before = "hi"
    after = "jumbo"
    newls = [after if x == before else x for x in ls]
    print(ls)
    print(newls)


def test4():
    with open('Glove_RT.json') as f:
        glove_vectors = json.load(f)

    res = generateBugs("happy", glove_vectors)
    print(res)


# test4()

def test5():
    l1 = [[1,2,3],[4,5,6],[7,8,9], [2,2,2], [3,6,9]]
    l1y = [0,0,1,1,0]
    nl = np.array(l1)
    print(nl)

    nl = np.concatenate(nl,np.array([0,0,0]))
    print(nl)


    X_train,x_test,y_train,y_test = train_test_split(l1, l1y, test_size=0.2)
    print(X_train)
    print(x_test)
    print(y_train)
    print(y_test)
    
    
    # print(nl)






test5()



