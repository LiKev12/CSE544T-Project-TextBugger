from __future__ import unicode_literals, print_function
from spacy.lang.en import English # updated
import json
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import pprint
import nltk
import time

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras import Input
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()
from keras import backend as k

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nltk.tokenize.treebank import TreebankWordDetokenizer




def make_LSTM(dataset, num_epochs):
    data = pickle.load( open( "datasets/{}/{}_embeddings.p".format(dataset, dataset), "rb" ) )
    emb_map = pickle.load( open( "datasets/{}/{}_embed_map.p".format(dataset, dataset), "rb" ) )
    vocab_size = len(list(emb_map['w2i'].keys()))
    print('Vocab size is {}'.format(vocab_size))

    ## Train
    x_train_pos = data['train']['pos']
    x_train_neg = data['train']['neg'][0:len(x_train_pos)]
    print("POS: {}".format(len(x_train_pos)))
    print("NEG: {}".format(len(x_train_neg)))
    x_train = []
    x_train.extend(x_train_pos)
    x_train.extend(x_train_neg)
    y_train = [1 for i in range(len(x_train_pos))]
    y_train.extend([0 for i in range(len(x_train_neg))])
    x_train = np.array(x_train, dtype='float')
    y_train = np.array(y_train, dtype='float')

    ## Test
    x_test_pos = data['test']['pos']
    x_test_neg = data['test']['neg'][0:len(x_test_pos)]
    x_test = []
    x_test.extend(x_test_pos)
    x_test.extend(x_test_neg)
    y_test = [1 for i in range(len(x_test_pos))]
    y_test.extend([0 for i in range(len(x_test_neg))])
    x_test = np.array(x_test, dtype='float')
    y_test = np.array(y_test, dtype='float')

    ## Shuffle
    x_train,y_train = shuffle(x_train, y_train)
    x_test,y_test = shuffle(x_test,y_test)

    ## Model
    hidden_size = 32

    sl_model = Sequential()
    sl_model.add(Embedding(vocab_size, hidden_size))
    sl_model.add(LSTM(hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2))
    sl_model.add(Dense(1, activation='sigmoid'))
    sl_model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

    sl_model.fit(x_train, y_train, batch_size = 128, epochs = num_epochs, validation_data = (x_test, y_test), shuffle=True)
    loss, acc = sl_model.evaluate(x_test, y_test)

    print('Single layer model -- ACC {} -- LOSS {}'.format(acc,loss))

    print('{} model done!'.format(dataset))

    pickle.dump(sl_model, open( "models/LSTM_TCD_{}.p".format(dataset), "wb" ))

    return




def get_cnn(dataset, epochs):
    data = pickle.load( open( "datasets/{}/{}_embeddings.p".format(dataset, dataset), "rb" ) )
    emb_map = pickle.load( open( "datasets/{}/{}_embed_map.p".format(dataset, dataset), "rb" ) )
    vocab_size = len(list(emb_map['w2i'].keys()))
    print('Vocab size is {}'.format(vocab_size))

    ## Train
    x_train_pos = data['train']['pos']
    x_train_neg = data['train']['neg'][0:len(x_train_pos)]
    x_train = []
    x_train.extend(x_train_pos)
    x_train.extend(x_train_neg)
    y_train = [1 for i in range(len(x_train_pos))]
    y_train.extend([0 for i in range(len(x_train_neg))])
    x_train = np.array(x_train, dtype='float')
    y_train = np.array(y_train, dtype='float')

    ## Test
    x_test_pos = data['test']['pos']
    x_test_neg = data['test']['neg'][0:len(x_test_pos)]
    x_test = []
    x_test.extend(x_test_pos)
    x_test.extend(x_test_neg)
    y_test = [1 for i in range(len(x_test_pos))]
    y_test.extend([0 for i in range(len(x_test_neg))])
    x_test = np.array(x_test, dtype='float')
    y_test = np.array(y_test, dtype='float')


    ## Shuffle
    x_train,y_train = shuffle(x_train, y_train)
    x_test,y_test = shuffle(x_test,y_test)

    ## Model
    max_len = x_train.shape[1]
    batch_size = 32
    embedding_dims=10
    filters=16
    kernel_size=3
    hidden_dims=250

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dims, input_length=max_len))

    model.add(Dropout(0.5))
    model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data=(x_test, y_test))

    loss, acc = model.evaluate(x_test, y_test)
    print('CNN Model -- ACC {} -- LOSS {}'.format(acc,loss))
    print('{} model done!'.format(dataset))


    pickle.dump(model, open( "models/CNN/CNN_TCD_{}.p".format(dataset), "wb" ))




def make_LR(dataset):
    data = pickle.load( open( "datasets/{}/{}_vectors.p".format(dataset, dataset), "rb" ) )

    ## Train
    x_train_pos = data['train']['pos']
    x_train_neg = data['train']['neg'][0:len(x_train_pos)]
    x_train = []
    x_train.extend(x_train_pos)
    x_train.extend(x_train_neg)
    y_train = [1 for i in range(len(x_train_pos))]
    y_train.extend([0 for i in range(len(x_train_neg))])
    x_train = np.array(x_train, dtype='float')
    y_train = np.array(y_train, dtype='float')

    ## Test
    x_test_pos = data['test']['pos']
    x_test_neg = data['test']['neg'][0:len(x_test_pos)]
    x_test = []
    x_test.extend(x_test_pos)
    x_test.extend(x_test_neg)
    y_test = [1 for i in range(len(x_test_pos))]
    y_test.extend([0 for i in range(len(x_test_neg))])
    x_test = np.array(x_test, dtype='float')
    y_test = np.array(y_test, dtype='float')

    ## Shuffle
    x_train,y_train = shuffle(x_train, y_train)
    x_test,y_test = shuffle(x_test,y_test)

    model = LogisticRegression(random_state=42).fit(x_train,y_train)
    acc = model.score(x_test,y_test)

    print("ACCURACY: {}".format(acc))
    pickle.dump(model, open( "models/LR/LR_TCD_{}.p".format(dataset), "wb" ))


# make_LSTM('Kaggle',2) # 86.53%
# get_cnn('Kaggle',2) # 87.40%
# make_LR('Kaggle') # 87.51
