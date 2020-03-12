from __future__ import print_function

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
from sklearn.preprocessing import StandardScaler
from feature_transform import transform_to_feature_vector


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow import keras
from sklearn.utils import shuffle


## IMDB Example Keras

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb



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

    with open(imdb_path, 'rb') as f:
        imdb_tokens = pickle.load(f)

    imdb_glove = {}
    for key1 in imdb_tokens:
        for key2 in imdb_tokens[key1]:
            sentences = imdb_tokens[key1][key2]
            for sentence in sentences:
                for word in sentence:
                    if word in glove_vectors:
                        imdb_glove[word] = glove_vectors[word]

    with open('Glove_IMDB.json', 'w') as f:
        json.dump(imdb_glove, f)


    ## Rotten Tomatoes
    # rt_path = "data/RT_text_tokenized.p"

    # with open(rt_path, 'rb') as f:
    #     rt_tokens = pickle.load(f)

    # rt_glove = {}
    # for key1 in rt_tokens:
    #     sentences = rt_tokens[key1]
    #     for sentence in sentences:
    #         for word in sentence:
    #             if word in glove_vectors:
    #                 rt_glove[word] = glove_vectors[word]

    # with open('Glove_RT.json', 'w') as f:
    #     json.dump(rt_glove, f)





get_glove_IMDB()









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






# test5()




def testLSTM():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train/ 255.0
    x_test = x_test/ 255.0

    # print(x_train.shape)
    # print(x_train.shape[1:])

    model = Sequential()

    model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    model.compile(loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3, validation_data=(x_test,y_test))




# testLSTM()

def testLSTM_RT():
    ## Load Data RT


    with open('Glove_RT.json') as f:
        glove_vectors = json.load(f)


    with open('data/RT_tokenized_vectors_TRAIN.p', 'rb') as fp:
        data_rt = pickle.load(fp)

    with open('data/RT_tokenized_vectors_TEST.p', 'rb') as fp:
        tests = pickle.load(fp)

    Xs = np.concatenate((data_rt['pos'], data_rt['neg']), axis=0)
    # Xs = d
    ys = [1] * len(data_rt['pos'])  # 2665
    ys_neg = [0] * len(data_rt['neg'])
    ys.extend(ys_neg) # 2665
    # Xs,ys = shuffle(Xs,ys)

    X_test = np.concatenate((tests['pos'], tests['neg']), axis=0)
    y_test = [1] * len(tests['pos'])
    y_test.extend([0] * len(tests['neg']))

    # X_test, y_test = shuffle(X_test, y_test)



    for i in range(0,len(Xs)):
        Xs[i] = np.array(Xs[i]).transpose()
    
    for i in range(0, len(X_test)):
        X_test[i] = np.array(X_test[i]).transpose()

    ys = np.array(ys)
    y_test = np.array(y_test)

    a = np.zeros((len(Xs),300,1))
    b = np.zeros((len(X_test),300,1))

    for i in range(0,len(a)):
        a[i] = Xs[i]
        b[i] = X_test[i]

    print(a.shape)

    Xs = a
    X_test = b
    print(Xs.shape)
    print(Xs.shape[1:])
    print(X_test.shape)
    print(X_test.shape[1:])


    model = Sequential()

    model.add(LSTM(128, input_shape=(Xs.shape[1:]), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    model.compile(loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    model.fit(Xs, ys, epochs=3, validation_data=(X_test, y_test))


# testLSTM_RT()

# l1 = [[1,2,3,4,5],[2,3,4,5,6]]
# print(np.array(l1)[1:].shape)


def testLSTM_IMDB():
    ## Load Data IMDB
    with open('data/IMDB_text_tokenized.p') as f:
        data_imdb = pickle.load(f)


# testLSTM_IMDB()





def test_example_LSTM():
    max_features = 20000
    # cut texts after this number of words (among top max_features most common words)
    maxlen = 80
    batch_size = 32

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print(y_train.shape)
    print(y_test.shape)
    return
    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=15,
            validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)




def test_example_LSTM2():
    with open('data/RT_vectors_TRAIN.p', 'rb') as fp:
        train_vectors = pickle.load(fp)
    with open('data/RT_vectors_TEST.p', 'rb') as fp:
        test_vectors = pickle.load(fp)


    train_pos = train_vectors['pos']
    train_neg = train_vectors['neg']
    test_pos = test_vectors['pos']
    test_neg = test_vectors['neg']

    # train = np.concatenate((train_pos,train_neg), axis=0)
    # test = np.concatenate((test_pos, test_neg), axis=0)


    train = []
    test = []
    for i in range(0,len(train_pos)):
        train.append(train_pos[i])
    for i in range(0,len(train_neg)):
        train.append(train_neg[i])
    for i in range(0,len(test_pos)):
        test.append(test_pos[i])
    for i in range(0,len(test_neg)):
        test.append(test_neg[i])


    x_train = np.array(train)
    x_test = np.array(test)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = [1] * len(train_pos)
    y_train.extend([0] * len(train_neg))
    y_test = [1] * len(test_pos)
    y_test.extend([0] * len(test_neg))
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(len(x_train))
    print(len(y_train))
    print(len(x_test))
    print(len(y_test))


    max_features = 20000
    batch_size = 32

    x_train = np.reshape(x_train, x_train.shape + (1,))
    x_test = np.reshape(x_test, x_test.shape+(1,))

    model = Sequential()
    # model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    # model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

    model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train, 
        batch_size=batch_size,
        epochs=15,
        validation_data=(x_test, y_test))
    score,acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


    x_train = np.reshape(x_train, x_train.shape + (1,))
    x_test = np.reshape(x_test, x_test.shape+(1,))
    return

# test_example_LSTM()
# test_example_LSTM2()







def test_myself():
    with open('Glove_RT.json') as f:
        glove_vectors = json.load(f)
    
    ## just to test...
    with open('data/RT_tokenized_TRAIN.p', 'rb') as fp:
        train_words = pickle.load(fp)
    
    ## just to test...
    with open('data/RT_tokenized_TEST.p', 'rb') as fp:
        test_words = pickle.load(fp)



    pos_train = train_words['pos']
    neg_train = train_words['neg']

    pos_test = test_words['pos']
    neg_test = test_words['neg']


    for i in range(0,len(pos_train)):
        [point] = transform_to_feature_vector(pos_train[i],glove_vectors)
        pos_train[i] = point
    for i in range(0,len(neg_train)):
        [point] = transform_to_feature_vector(neg_train[i],glove_vectors)
        neg_train[i] = point
    for i in range(0,len(pos_test)):
        [point] = transform_to_feature_vector(pos_test[i],glove_vectors)
        pos_test[i] = point
    for i in range(0,len(neg_test)):
        [point] = transform_to_feature_vector(neg_test[i],glove_vectors)
        neg_test[i] = point

    train_words['pos'] = pos_train
    train_words['neg'] = neg_train
    test_words['pos'] = pos_test
    test_words['neg'] = neg_test


    print(len(pos_train))
    print(len(neg_train))
    print(len(pos_test))
    print(len(neg_test))

    with open('data/RT_vectors_TRAIN.p', 'wb') as fp:
        pickle.dump(train_words, fp)

    with open('data/RT_vectors_TEST.p', 'wb') as fp:
        pickle.dump(test_words, fp)


# test_myself()












def get_IMDB_Vectors():

    with open('Glove_IMDB.json') as f:
        glove_vectors = json.load(f)
    with open('data/IMDB_text_tokenized.p', 'rb') as fp:
        imdb_data = pickle.load(fp)

    for key1 in imdb_data:
        for key2 in imdb_data[key1]:
            for i in range(0,len(imdb_data[key1][key2])):
                feature = transform_to_feature_vector(imdb_data[key1][key2][i], glove_vectors)
    


# get_IMDB_Vectors()