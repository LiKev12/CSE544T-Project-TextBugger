from __future__ import unicode_literals, print_function
#from spacy.lang.en import English # updated
import json
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import pprint
import nltk
import time
import pandas as pd

# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Embedding
# from keras.datasets import imdb
# from keras.preprocessing import sequence
# from keras.layers import Dense, Dropout, Activation
# from keras.layers import Embedding, LSTM
# from keras.layers import Conv1D, Flatten, MaxPooling1D
# from keras import Input
# # import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# from keras import backend as k

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nltk.tokenize.treebank import TreebankWordDetokenizer
from textbugger_utils import *



def get_vector_same_len(doc,max_len):
    
    doc_size = len(doc)

    if (doc_size > max_len):
        res = doc[0:max_len]
        return res
    elif (doc_size < max_len):
        diff = max_len - doc_size
        for i in range(0,diff):
            doc.append(random.randint(1,10))
        return doc
    else:
        return doc

def transform_to_feature_vector(tokens, glove_vectors):
    vectors = []
    num_outliers = 0
    for token in tokens:
        if token in glove_vectors:
            vect = glove_vectors[token]
            vectors.append(vect)
        else:
            # sampling from the uniform distribution in [-0.1, 0.1]
            num_outliers += 1
            vect = [(random.random()/5)-0.1 for i in range(300)]
            vectors.append(vect)

    means = np.mean(vectors, axis=0)
    # if (num_outliers > 0):
    #     print("outliers: " + str(num_outliers))
    return means

def testLSTM():

    with open('Glove_IMDB.json') as f:
        glove_vectors = json.load(f)


    with open('data/IMDB_text_tokenized_lower.p','rb') as fp:
        data = pickle.load(fp)
    with open('data/IMDB_vectors.p','rb') as fp:
        data_v = pickle.load(fp)



# testLSTM()


def tutorial():
    max_words = 20000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_words)
    print("X_train length: {}".format(len(x_train)))
    print("X_test length: {}".format(len(x_train)))


    # words_to_index = imdb.get_word_index()
    # index_to_word = {v:k for k,v in word_to_index.items()}
    # print(x_train[0])
    # print(" ".join([index_to_word[x] for x in x_train[0]]))
    

    max_sequence_length = 180
    x_train = sequence.pad_sequences(x_train, maxlen=max_sequence_length, padding='post',truncating='post')
    x_test = sequence.pad_sequences(x_test, maxlen=max_sequence_length, padding='post',truncating='post')


    print("X_train shape: {}".format(x_train.shape))
    print("X_test shape: {}".format(x_test.shape))
    print(x_train[0])

    print(y_test.shape)

    return
    hidden_size = 32

    sl_model = Sequential()
    sl_model.add(Embedding(max_words, hidden_size))
    sl_model.add(LSTM(hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2))
    sl_model.add(Dense(1, activation='sigmoid'))
    sl_model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    
    epochs = 3

    sl_model.fit(x_train, y_train, epochs = epochs, shuffle=True)
    loss, acc = sl_model.evaluate(x_test, y_test)

    print('Single layer model -- ACC {} -- LOSS {}'.format(acc,loss))




# tutorial()










def myLSTM():

    data = pickle.load( open( "datasets/IMDB_vectors.p", "rb" ) )

    ## Train
    x_train = data['train']['pos']
    x_train.extend(data['train']['neg'])
    y_train = [1 for i in range(12500)]
    y_train.extend([0 for i in range(12500)])
    x_train, y_train= shuffle(x_train, y_train)
    x_train = np.array(x_train, dtype='float')
    y_train = np.array(y_train, dtype='float')

    ## Test
    x_test = data['test']['pos']
    x_test.extend(data['test']['neg'])
    y_test = [1 for i in range(12500)]
    y_test.extend([0 for i in range(12500)])
    x_test, y_test = shuffle(x_test, y_test)
    x_test = np.array(x_test, dtype='float')
    y_test = np.array(y_test, dtype='float')

    ## Model
    hidden_size = 32
    sl_model = Sequential()
    sl_model.add(Embedding(1000, hidden_size))
    sl_model.add(LSTM(hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2))
    sl_model.add(Dense(1, activation='sigmoid'))
    sl_model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])


    epochs = 3

    sl_model.fit(x_train, y_train, epochs = epochs, shuffle=True)
    loss, acc = sl_model.evaluate(x_test, y_test)

    print('Single layer model -- ACC {} -- LOSS {}'.format(acc,loss))



# myLSTM()





output_path = "datasets/IMDB_embeddings.p"
def get_embed_data(dataset,data,output_path):

    emb = {}
    emb_map = pickle.load( open( "datasets/embedding_word_to_index.p", "rb" ) )
    for key1 in data:
        emb[key1] = {}
        for key2 in data[key1]:
            emb[key1][key2] = []
            for document in data[key1][key2]:
                document = chunk_input(document, dataset)
                vector = get_single_embed(document, emb_map)
                emb[key1][key2].append(vector)
    
    pickle.dump( emb, open(output_path, "wb"))
    return emb

def get_single_embed(doc,emb_map):
    res = []
    for word in doc:
        res.append(emb_map[word])
    return res






def get_embed_same_len():
    data = pickle.load( open( "datasets/IMDB_embeddings.p", "rb" ) )

    max_len = 200

    res = {}
    for key1 in data:
        res[key1] = {}
        for key2 in data[key1]:
            res[key1][key2] = []
            for doc in data[key1][key2]:
                norm = get_vector_same_len(doc,max_len)
                res[key1][key2].append(norm)
                if (len(norm) != max_len):
                    print("HOW")

    pickle.dump( res, open( "datasets/IMDB_embeddings_same_len.p", "wb" ))


# get_embed_same_len()




def newLSTM():
    data = pickle.load( open( "datasets/IMDB_embeddings_same_len.p", "rb" ) )

    emb_map = pickle.load( open( "datasets/embedding_word_to_index.p", "rb" ) )
    vocab_size = len(list(emb_map.keys()))
    print('Vocab size is {}'.format(vocab_size))

    ## Train
    x_train = data['train']['pos']
    x_train.extend(data['train']['neg'])
    y_train = [1 for i in range(12500)]
    y_train.extend([0 for i in range(12500)])
    x_train, y_train= shuffle(x_train, y_train)
    x_train = np.array(x_train, dtype='float')
    y_train = np.array(y_train, dtype='float')

    ## Test
    x_test = data['test']['pos']
    x_test.extend(data['test']['neg'])
    y_test = [1 for i in range(12500)]
    y_test.extend([0 for i in range(12500)])
    x_test, y_test = shuffle(x_test, y_test)
    x_test = np.array(x_test, dtype='float')
    y_test = np.array(y_test, dtype='float')

    x_train,y_train = shuffle(x_train, y_train)
    x_test,y_test = shuffle(x_test,y_test)

    # ## Model
    hidden_size = 32
    epochs = 50

    sl_model = Sequential()
    sl_model.add(Embedding(vocab_size, hidden_size))
    sl_model.add(LSTM(hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2))
    sl_model.add(Dense(1, activation='sigmoid'))
    sl_model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])



    sl_model.fit(x_train, y_train, epochs = epochs, shuffle=True)
    loss, acc = sl_model.evaluate(x_test, y_test)

    print('Single layer model -- ACC {} -- LOSS {}'.format(acc,loss))

    pickle.dump(sl_model, open( "models/LSTM_SA_IMDB.p", "wb" ))

    return




# newLSTM()


def keras_model():
    data = pickle.load( open( "datasets/IMDB_embeddings_same_len.p", "rb" ) )

    emb_map = pickle.load( open( "datasets/embedding_word_to_index.p", "rb" ) )
    vocab_size = len(list(emb_map.keys()))
    print('Vocab size is {}'.format(vocab_size))

    ## Train
    x_train = data['train']['pos']
    x_train.extend(data['train']['neg'])
    y_train = [1 for i in range(12500)]
    y_train.extend([0 for i in range(12500)])
    x_train, y_train= shuffle(x_train, y_train)
    x_train = np.array(x_train, dtype='float')
    y_train = np.array(y_train, dtype='float')

    ## Test
    x_test = data['test']['pos']
    x_test.extend(data['test']['neg'])
    y_test = [1 for i in range(12500)]
    y_test.extend([0 for i in range(12500)])
    x_test, y_test = shuffle(x_test, y_test)
    x_test = np.array(x_test, dtype='float')
    y_test = np.array(y_test, dtype='float')


    x_test,y_test = shuffle(x_test,y_test)


    batch_size = 32
    print('Build model...')
    model = Sequential()
    model.add(Embedding(vocab_size, 128))
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







# keras_model()









data_path = "datasets/IMDB_text_tokenized_lower.p"
def get_embedding(data_path):

    data = pickle.load(open(data_path, "rb"))


    emb = {}
    idx = 0
    for key1 in data:
        for key2 in data[key1]:
            for document in data[key1][key2]:
                for word in document:
                    if (word not in emb): 
                        emb[word] = 1
                    else:
                        emb[word] += 1


    emb_sorted = {k: v for k, v in sorted(emb.items(), key=lambda item: -item[1])}

    idx = 0
    for key in emb_sorted:
        emb_sorted[key] = idx
        idx += 1

    idx = 0

    pickle.dump( emb_sorted, open( "datasets/embedding_word_to_index.p", "wb" ))


    rev = {v:k for k,v in emb_sorted.items()}

    pickle.dump(rev, open("datasets/embedding_index_to_word.p","wb"))

    return emb_sorted.keys()



# get_embedding()







def lstm_rt():
    data = pickle.load( open( "datasets/RT/RT_tokens.p", "rb" ) )


    lens = []
    for key1 in data:
        for key2 in data[key1]:
            for doc in data[key1][key2]:
                lens.append(len(doc))


    lens = sorted(lens)
    avg_len = np.mean(lens)
    med_len = np.median(lens)


    print(avg_len)
    print(med_len)








# lstm_rt()

def create_rt_tokens():
    data = pickle.load( open( "datasets/RT/RT_tokens.p", "rb" ) )

    emb = {}
    idx = 0
    for key1 in data:
        for key2 in data[key1]:
            for document in data[key1][key2]:
                for word in document:
                    if (word not in emb): 
                        emb[word] = 1
                    else:
                        emb[word] += 1

    emb_sorted = {k: v for k, v in sorted(emb.items(), key=lambda item: -item[1])}

    idx = 0
    for key in emb_sorted:
        emb_sorted[key] = idx
        idx += 1

    idx = 0

    pickle.dump( emb_sorted, open( "datasets/RT/RT_embed_map_word_to_index.p", "wb" ))
    print(emb_sorted)

    rev = {v:k for k,v in emb_sorted.items()}

    pickle.dump(rev, open("datasets/RT/RT_embed_map_index_to_word.p","wb"))
    print(rev)


# create_rt_tokens()










def format_rt():
    data = pickle.load( open( "datasets/RT/RT_embeddings.p", "rb" ) )
    max_len = 20

    res = {}
    for key1 in data:
        res[key1] = {}
        for key2 in data[key1]:
            res[key1][key2] = []
            for doc in data[key1][key2]:
                norm = get_vector_same_len(doc,max_len)
                res[key1][key2].append(norm)
                if (len(norm) != max_len):
                    print("HOW")

    pickle.dump( res, open( "datasets/RT/RT_embeddings2.p", "wb" ))


def create_embeds():
    data = pickle.load( open( "datasets/RT/RT_tokens.p", "rb" ) )
    emb_map = pickle.load( open( "datasets/embed_map_word_to_index.p", "rb" ) )

    emb = {}
    for key1 in data:
        emb[key1] = {}
        for key2 in data[key1]:
            emb[key1][key2] = []
            for doc in data[key1][key2]:
                vector = get_single_embed(doc, emb_map)
                emb[key1][key2].append(vector)

    pickle.dump( emb, open( "datasets/RT/RT_embeddings.p", "wb" ))


# create_embeds()




def inspect2():
    data = pickle.load( open( "datasets/RT/RT_embeddings.p", "rb" ) )
    print(data['train']['pos'][0])

# inspect2()



def testf1():
    emb_map = pickle.load( open( "datasets/RT/RT_embed_word_to_index.p", "rb" ) )
    print(emb_map)



# testf1()



# def get_embed_data(data):
    
#     emb = {}
#     emb_map = pickle.load( open( "datasets/embedding_word_to_index.p", "rb" ) )
#     for key1 in data:
#         emb[key1] = {}
#         for key2 in data[key1]:
#             emb[key1][key2] = []
#             for document in data[key1][key2]:
#                 vector = get_single_embed(document, emb_map)
#                 emb[key1][key2].append(vector)
    
#     pickle.dump( emb, open( "datasets/IMDB_embeddings.p", "wb" ))
#     return emb

# def get_single_embed(doc,emb_map):
#     res = []
#     for word in doc:
#         res.append(emb_map[word])
#     return res










def combine():
    data_rt = pickle.load( open( "datasets/RT/RT_tokens.p", "rb" ) )
    data_imdb = pickle.load( open( "datasets/IMDB/IMDB_tokens.p", "rb" ) )

    emb = {}
    idx = 0
    for key1 in data_rt:
        for key2 in data_rt[key1]:
            for document in data_rt[key1][key2]:
                for word in document:
                    if (word not in emb): 
                        emb[word] = 1
                    else:
                        emb[word] += 1

    for key1 in data_imdb:
        for key2 in data_imdb[key1]:
            for document in data_imdb[key1][key2]:
                for word in document:
                    if (word not in emb): 
                        emb[word] = 1
                    else:
                        emb[word] += 1


    emb_sorted = {k: v for k, v in sorted(emb.items(), key=lambda item: -item[1])}
    print(len(list(emb_sorted.keys())))
    idx = 0
    for key in emb_sorted:
        emb_sorted[key] = idx
        idx += 1

    pickle.dump( emb_sorted, open( "datasets/embed_map_word_to_index.p", "wb" ))

    rev = {v:k for k,v in emb_sorted.items()}
    pickle.dump(rev, open("datasets/embed_map_index_to_word.p","wb"))



# combine()


def inspect():
    data = pickle.load( open( "datasets/embed_map_word_to_index.p", "rb" ) )
    

    idx = 0
    for key in data:
        print(key + " | " + str(data[key]))
        idx += 1
        if idx > 20:
            break






# inspect()









#### MODELING ---------------------------------------------------



def make_LSTM(dataset, num_epochs):
    data = pickle.load( open( "datasets/{}/{}_embeddings.p".format(dataset, dataset), "rb" ) )
    emb_map = pickle.load( open( "datasets/embed_map.p", "rb" ) )
    vocab_size = len(list(emb_map['w2i'].keys()))
    print('Vocab size is {}'.format(vocab_size))

    ## Train
    x_train_pos = data['train']['pos']
    x_train_neg = data['train']['neg']
    x_train = []
    x_train.extend(x_train_pos)
    x_train.extend(x_train_neg)
    y_train = [1 for i in range(len(x_train_pos))]
    y_train.extend([0 for i in range(len(x_train_neg))])
    x_train = np.array(x_train, dtype='float')
    y_train = np.array(y_train, dtype='float')

    ## Test
    x_test_pos = data['test']['pos']
    x_test_neg = data['test']['neg']
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

    sl_model.fit(x_train, y_train, epochs = num_epochs, shuffle=True)
    loss, acc = sl_model.evaluate(x_test, y_test)

    print('Single layer model -- ACC {} -- LOSS {}'.format(acc,loss))

    print('{} model done!'.format(dataset))

    pickle.dump(sl_model, open( "models/LSTM_SA_{}.p".format(dataset), "wb" ))

    return


# make_LSTM('IMDB',6) # 76.32%
# make_LSTM('RT',3) # 72.3%









def make_LR(dataset):
    data = pickle.load( open( "datasets/{}/{}_vectors.p".format(dataset, dataset), "rb" ) )

    ## Train
    x_train_pos = data['train']['pos']
    x_train_neg = data['train']['neg']
    x_train = []
    x_train.extend(x_train_pos)
    x_train.extend(x_train_neg)
    y_train = [1 for i in range(len(x_train_pos))]
    y_train.extend([0 for i in range(len(x_train_neg))])
    x_train = np.array(x_train, dtype='float')
    y_train = np.array(y_train, dtype='float')

    ## Test
    x_test_pos = data['test']['pos']
    x_test_neg = data['test']['neg']
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
    print(acc)
    # preds = model.predict_proba(x_test)

    # pickle.dump(model, open( "models/LR/LR_SA_{}.p".format(dataset), "wb" ))



# make_LR('RT') # 77.42%
# make_LR('IMDB') # 84.96%%








def check_IMDB(dataset):
    tokens = pickle.load( open( "datasets/{}/{}_tokens.p".format(dataset, dataset), "rb" ) )
    vectors = pickle.load( open( "datasets/{}/{}_vectors.p".format(dataset, dataset), "rb" ) )

    glove_vectors = json.load( open( "Glove_IMDB.json", "rb" ) )

    train_pos_A = transform_to_feature_vector(tokens['train']['pos'][0], glove_vectors)
    train_pos_B = vectors['train']['pos'][0]
    test_pos_A = transform_to_feature_vector(tokens['test']['pos'][0], glove_vectors)
    test_pos_B = vectors['test']['pos'][0]

    train_neg_A = transform_to_feature_vector(tokens['train']['neg'][0], glove_vectors)
    train_neg_B = vectors['train']['neg'][0]
    test_neg_A = transform_to_feature_vector(tokens['test']['neg'][0], glove_vectors)
    test_neg_B = vectors['test']['neg'][0]



    # print(' '.join(tokens['train']['pos'][0]))
    # print()
    # print(' '.join(tokens['test']['pos'][0]))
    # print()
    # print(' '.join(random.choice(tokens['train']['neg'])))
    # print()
    # print(' '.join(random.choice(tokens['test']['neg'])))
    # print()

    # return

    fs = list(range(300))
    fig = plt.figure()

    plt.subplot(4, 2, 1)
    plt.scatter(fs, test_pos_A,s=2, color='orange')

    plt.subplot(4, 2, 2)
    plt.scatter(fs, train_pos_B,s=2, color='blue')

    plt.subplot(4, 2, 3)
    plt.scatter(fs, test_pos_A,s=2, color='orange')

    plt.subplot(4, 2, 4)
    plt.scatter(fs, test_pos_B,s=2, color='blue')

    plt.subplot(4, 2, 5)
    plt.scatter(fs, train_neg_A,s=2, color='orange')

    plt.subplot(4, 2, 6)
    plt.scatter(fs, train_neg_B,s=2, color='blue')

    plt.subplot(4, 2, 7)
    plt.scatter(fs, test_neg_A,s=2, color='orange')

    plt.subplot(4, 2, 8)
    plt.scatter(fs, test_neg_B,s=2, color='blue')

    plt.show()



# check_IMDB('IMDB')


def test_original():
    # model = pickle.load( open( "Sentiment_Analysis/White_Box/Models/LogisticRegression_IMDB.pkl", "rb" ) )
    # model = pickle.load( open( "models/LR/.pkl", "rb" ) )

    # data = pickle.load( open( "datasets/{}/{}_vectors.p".format('IMDB', 'IMDB'), "rb" ) )
    data = pickle.load( open( "data/IMDB_vectors.p", "rb" ) )

    ## Train
    x_train_pos = data['train']['pos']
    x_train_neg = data['train']['neg']
    x_train = []
    x_train.extend(x_train_pos)
    x_train.extend(x_train_neg)
    y_train = [1 for i in range(len(x_train_pos))]
    y_train.extend([0 for i in range(len(x_train_neg))])
    x_train = np.array(x_train, dtype='float')
    y_train = np.array(y_train, dtype='float')

    ## Test
    x_test_pos = data['test']['pos']
    x_test_neg = data['test']['neg']
    x_test = []
    x_test.extend(x_test_pos)
    x_test.extend(x_test_neg)
    y_test = [1 for i in range(len(x_test_pos))]
    y_test.extend([0 for i in range(len(x_test_neg))])
    x_test = np.array(x_test, dtype='float')
    y_test = np.array(y_test, dtype='float')

    model = LogisticRegression(random_state=42).fit(x_train,y_train)


    acc = model.score(x_test,y_test)
    print(acc)



# test_original()






def compare_vects():
    ori = pickle.load( open( "data/IMDB_vectors.p", "rb" ) )
    new = pickle.load( open( "datasets/IMDB/IMDB_vectors.p","rb"))


    print(ori['train']['pos'][0])
    print(new['train']['pos'][0])
    print(ori['train']['neg'][0])
    print(new['train']['neg'][0])
    print(ori['test']['pos'][0])
    print(new['test']['pos'][0])
    print(ori['test']['neg'][0])
    print(new['test']['neg'][0])


    fs = list(range(300))
    fig = plt.figure()

    plt.subplot(4, 2, 1)
    plt.scatter(fs, ori['train']['pos'][0],s=2, color='orange')

    plt.subplot(4, 2, 2)
    plt.scatter(fs, new['train']['pos'][0],s=2, color='blue')

    plt.subplot(4, 2, 3)
    plt.scatter(fs, ori['train']['neg'][0],s=2, color='orange')

    plt.subplot(4, 2, 4)
    plt.scatter(fs, new['train']['neg'][0],s=2, color='blue')

    plt.subplot(4, 2, 5)
    plt.scatter(fs, ori['test']['pos'][0],s=2, color='orange')

    plt.subplot(4, 2, 6)
    plt.scatter(fs, new['test']['pos'][0],s=2, color='blue')

    plt.subplot(4, 2, 7)
    plt.scatter(fs, ori['test']['neg'][0],s=2, color='orange')

    plt.subplot(4, 2, 8)
    plt.scatter(fs, new['test']['neg'][0],s=2, color='blue')

    plt.show()





def create_new():
    tokens = pickle.load( open( "datasets/IMDB/IMDB_text_tokenized.p","rb"))

    tokensB = pickle.load( open( "datasets/IMDB/IMDB_tokens.p","rb"))

    glove_vectors = json.load( open( "glove_final.json","rb"))
    


    # # print(tokens['train']['pos'][0])
    # # print(tokensB['train']['pos'][0])
    # # print()
    # # print(tokens['train']['neg'][0])
    # # print(tokensB['train']['neg'][0])
    # # print()
    # # print(tokens['test']['pos'][0])
    # # print(tokensB['test']['pos'][0])
    # # print()
    # print(tokens['test']['neg'][0])
    # print(tokensB['test']['neg'][0])


    for key1 in tokens:
        for key2 in tokens[key1]:
            for doc in tokens[key1][key2]:
                vect = transform_to_feature_vector(doc, glove_vectors)
                break
            break
        break

    for key1 in tokensB:
        for key2 in tokensB[key1]:
            for doc in tokensB[key1][key2]:
                vect = transform_to_feature_vector(doc, glove_vectors)
                return


# create_new()





def create_new_vector():
    glove_vectors = json.load( open( "glove_final.json","rb"))
    data = pickle.load( open( "datasets/IMDB/IMDB_tokens.p","rb"))
    


    res = {}
    for key1 in data:
        res[key1] = {}
        for key2 in data[key1]:
            res[key1][key2] = []

    print(res)

    for key1 in data:
        for key2 in data[key1]:
            vects = []
            for doc in data[key1][key2]:
                vect = transform_to_feature_vector(doc, glove_vectors)
                vects.append(vect)
            res[key1][key2] = vects


    pickle.dump(res, open( "datasets/imdb_vectors_again.p", "wb" ))

# create_new_vector()


def check_goods():
    data = pickle.load( open( "datasets/imdb_vectors_again.p","rb"))


    # print(res)
    




# check_goods()





def check_lstm():

    model = pickle.load( open( "models/LSTM/LSTM_SA_IMDB.p","rb"))
    data = pickle.load( open( "datasets/IMDB/IMDB_embeddings.p", "rb" ) )
    emb_map = pickle.load( open( "datasets/embed_map.p", "rb" ) )
    vocab_size = len(list(emb_map['w2i'].keys()))
    print('Vocab size is {}'.format(vocab_size))

    # ## Train
    # x_train_pos = data['train']['pos']
    # x_train_neg = data['train']['neg']
    # x_train = []
    # x_train.extend(x_train_pos)
    # x_train.extend(x_train_neg)
    # y_train = [1 for i in range(len(x_train_pos))]
    # y_train.extend([0 for i in range(len(x_train_neg))])
    # x_train = np.array(x_train, dtype='float')
    # y_train = np.array(y_train, dtype='float')

    ## Test
    x_test_pos = data['test']['pos'][0:5]
    x_test_neg = data['test']['neg'][0:5]
    x_test = []
    x_test.extend(x_test_pos)
    x_test.extend(x_test_neg)
    y_test = [1 for i in range(len(x_test_pos))]
    y_test.extend([0 for i in range(len(x_test_neg))])
    x_test = np.array(x_test, dtype='float')
    y_test = np.array(y_test, dtype='float')


    acc = model.predict(x_test)

    res = model.evaluate(x_test,y_test)
    print(res)
    # print(acc)


    # [[0.02160342]
    #  [0.95032316]
    #  [0.9608328 ]
    #  [0.931748  ]
    #  [0.9716669 ]
    #  [0.02665349]
    #  [0.9218951 ]
    #  [0.8579215 ]
    #  [0.00486689]
    #  [0.01401571]]


# check_lstm()






def format_sizes():
    model = pickle.load( open( "models/LSTM/LSTM_SA_IMDB.p","rb"))
    data = pickle.load(open('datasets/IMDB/IMDB_embeddings.p','rb'))
    embed_map = pickle.load(open('datasets/embed_map.p','rb'))



    y_true = []
    y_pred = []
    for key1 in data:
        for key2 in data[key1]:
            idx = 0
            for i in range(len(data[key1][key2])):
                if key2 == 'pos':
                    y_true.append(1)
                else:
                    y_true.append(0)
                y_pred.append(model.predict(data[key1][key2][i])[0][0])
                idx += 1
                if (idx > 50):
                    break

    y_true = np.array(y_true)
    y_pred = np.round(np.array(y_pred),0)
    # print(y_true)
    # print(y_pred)

    wrong = 0
    for i in range(len(y_pred)):
        if y_true[i] != y_pred[i]:
            wrong+= 1
        
    total = len(y_true)
    print(wrong/total)

    vects_norm = {}
    # pickle.dump(vects_norm, open( "datasets/IMDB/IMDB_vectors_200.p", "wb" ))
    # pickle.dump(vects_norm, open( "datasets/IMDB/IMDB_tokens_200.p", "wb" ))



# format_sizes()

def get_bed(doc,embed_map,max_len):
    
    doc_size = len(doc)
    pre = []


    for word in doc:
        if (word in embed_map):
            pre.append(embed_map[word])
        else:
            pre.append(random.randint(1,10))



    if (doc_size > max_len):
        res = pre[0:max_len]
        return res
    elif (doc_size < max_len):
        diff = max_len - doc_size
        for i in range(0,diff):
            pre.append(random.randint(1,10))
        return pre
    else:
        return pre

def generate_new_embeddings(dataset,max_len):
    tokens = pickle.load( open( "datasets/{}/{}_tokens.p".format(dataset, dataset),"rb"))
    embed_map = pickle.load( open( "datasets/embed_map.p".format(dataset, dataset),"rb"))


    res = {}

    for key1 in tokens:
        res[key1] = {}
        for key2 in tokens[key1]:
            res[key1][key2] = []



    for key1 in tokens:
        for key2 in tokens[key1]:
            for doc in tokens[key1][key2]:
                point = get_bed(doc, embed_map['w2i'], max_len)
                res[key1][key2].append(point)

    pickle.dump(res, open( "datasets/{}/{}_embeddings_new.p".format(dataset, dataset), "wb" ))


# generate_new_embeddings('RT', 20)
# generate_new_embeddings('IMDB',200)

















def get_cnn(dataset, epochs):
    #data = pickle.load( open( "datasets/{}/{}_embeddings.p".format(dataset, dataset), "rb" ) )
    #emb_map = pickle.load( open( "datasets/embed_map.p", "rb" ) )
    data = pickle.load(open("datasets/{}/{}_idx_embeddings.p".format(dataset, dataset), "rb"))
    emb_map = pickle.load(open("datasets/{}/embedding_word_to_index.p".format(dataset), "rb"))
    #vocab_size = len(list(emb_map['w2i'].keys()))
    vocab_size = len(list(emb_map.keys()))
    print('Vocab size is {}'.format(vocab_size))

    ## Train
    x_train_pos = data['train']['pos']
    x_train_neg = data['train']['neg']
    x_train = []
    x_train.extend(x_train_pos)
    x_train.extend(x_train_neg)
    y_train = [1 for i in range(len(x_train_pos))]
    y_train.extend([0 for i in range(len(x_train_neg))])
    x_train, y_train = shuffle(x_train, y_train)
    x_train = np.array(x_train, dtype='float')
    y_train = np.array(y_train, dtype='float')

    ## Val
    x_val_pos = data['val']['pos']
    x_val_neg = data['val']['neg']
    x_val = []
    x_val.extend(x_val_pos)
    x_val.extend(x_val_neg)
    y_val = [1 for i in range(len(x_val_pos))]
    y_val.extend([0 for i in range(len(x_val_neg))])
    x_val, y_val = shuffle(x_val, y_val)
    x_val = np.array(x_val, dtype='float')
    y_val = np.array(y_val, dtype='float')

    ## Test
    x_test_pos = data['test']['pos']
    x_test_neg = data['test']['neg']
    x_test = []
    x_test.extend(x_test_pos)
    x_test.extend(x_test_neg)
    y_test = [1 for i in range(len(x_test_pos))]
    y_test.extend([0 for i in range(len(x_test_neg))])
    x_test, y_test = shuffle(x_test, y_test)
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

    model.save('models/CNN/CNN_SA_{}.h5'.format(dataset))
    # pickle.dump(model, open( "models/CNN/CNN_SA_{}.h5".format(dataset), "wb" ))


# get_cnn('RT',5) # 70.24%
# get_cnn('IMDB',2) # 85.20%





    


def see_tokens():

    idx = 0
    data = pickle.load( open( "datasets/IMDB/IMDB_tokens.p", "rb" ) )
    for key1 in data:
        for key2 in data[key1]:
            docs = data[key1][key2]
            for doc in docs:
                print(doc)
                idx += 1
                if idx > 5:
                    return

# see_tokens()



def debug_cnn(dataset):
    data = pickle.load( open( "datasets/{}/{}_embeddings.p".format(dataset, dataset), "rb" ) )
    model = pickle.load( open( "models/CNN/CNN_SA_{}.p".format(dataset), "rb" ) )
    emb_map = pickle.load( open( "datasets/embed_map.p", "rb" ) )
    vocab_size = len(list(emb_map['w2i'].keys()))
    print('Vocab size is {}'.format(vocab_size))

    ## Train
    x_train_pos = data['train']['pos']
    x_train_neg = data['train']['neg']
    x_train = []
    x_train.extend(x_train_pos)
    x_train.extend(x_train_neg)
    y_train = [1 for i in range(len(x_train_pos))]
    y_train.extend([0 for i in range(len(x_train_neg))])
    x_train = np.array(x_train, dtype='float')
    y_train = np.array(y_train, dtype='float')

    ## Test
    x_test_pos = data['test']['pos']
    x_test_neg = data['test']['neg']
    x_test = []
    x_test.extend(x_test_pos)
    x_test.extend(x_test_neg)
    y_test = [1 for i in range(len(x_test_pos))]
    y_test.extend([0 for i in range(len(x_test_neg))])
    x_test = np.array(x_test, dtype='float')
    y_test = np.array(y_test, dtype='float')

    # x_test = np.hstack((x_test, np.ones((x_test.shape[0],1))))
    # print(x_test.shape)


    print(x_test[0:5].shape)
    print(x_test[0].transpose().shape)

    loss, acc = model.evaluate(x_test[0:5], y_test[0:5])

    score = model.predict([[x_test[0]]])
    # score = model.predict(x_test[0:5])

    print(score)
    print(acc)





# debug_cnn('IMDB')



def try_fgsm(dataset):

    # model = pickle.load( open( "models/CNN/CNN_SA_{}.h5".format(dataset), "rb" ) )
    model = tf.keras.models.load_model("models/CNN/CNN_SA_{}.h5".format(dataset))
    data = pickle.load( open( "datasets/{}/{}_embeddings.p".format(dataset, dataset), "rb" ) )
    # emb_map = pickle.load( open( "datasets/embed_map.p", "rb" ) )
    # vocab_size = len(list(emb_map['w2i'].keys()))
    # print('Vocab size is {}'.format(vocab_size))

    ## Train
    x_train_pos = data['train']['pos']
    x_train_neg = data['train']['neg']
    x_train = []
    x_train.extend(x_train_pos)
    x_train.extend(x_train_neg)
    y_train = [1 for i in range(len(x_train_pos))]
    y_train.extend([0 for i in range(len(x_train_neg))])
    x_train = np.array(x_train, dtype='float')
    y_train = np.array(y_train, dtype='float')

    ## Test
    x_test_pos = data['test']['pos']
    x_test_neg = data['test']['neg']
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

    # ## Model
    # max_len = x_train.shape[1]
    # batch_size = 32
    # embedding_dims=10
    # filters=16
    # kernel_size=3
    # hidden_dims=250
    # epochs = 3

    # model = Sequential()
    # model.add(Embedding(vocab_size, embedding_dims, input_length=max_len))

    # model.add(Dropout(0.5))
    # model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu'))
    # model.add(MaxPooling1D())
    # model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu'))
    # model.add(MaxPooling1D())
    # model.add(Flatten())
    # model.add(Dense(hidden_dims, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    # model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    # model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data=(x_test, y_test))

    # loss, acc = model.evaluate(x_test, y_test)
    # print('CNN Model -- ACC {} -- LOSS {}'.format(acc,loss))
    # print('{} model done!'.format(dataset))
    # y_true = y_test
    # y_pred = model.predict(x_test)


    # print(y_true)
    # print(y_pred)

    # y_true = k.reshape(y_true, (-1, 2))
    # y_pred = k.reshape(y_pred, (-1, 2))


    print(model.summary())
    print(len(model.layers))
    get_dense_layer_output = k.function([model.layers[0].input],
                                  [model.layers[9].output])

    layer_output = get_dense_layer_output([x_train])[0]

    y_test = y_test.astype('float32')
    y_true = k.constant(y_test)
    y_pred = k.constant(layer_output)
    g = k.categorical_crossentropy(target=y_true, output=y_pred)
    # g = k.mean(k.categorical_crossentropy(target=y_true, output=y_pred))
    # grad_ce = k.gradients(g, model.input)
    # print(grad_ce)
    print(g)
    ce = k.eval(g)  # 'ce' for cross-entropy
    print(ce.shape)
    print(ce)
    # # output = func([model_input_array(s), true_labels])
    # func = k.function(model.inputs + [y_true], grad_ce)

    # # usage
    # output = func([x_test, true_labels])
    # print(output)
    # return


    return


    # y_true = Input(shape=labels_shape)
    ce = k.mean(k.categorical_crossentropy(y_true, model.output))
    grad_ce = k.gradients(ce, model.inputs)
    func = k.function(model.inputs + [y_true], grad_ce)
    output = func([model_input_array(s), true_labels])
    print(output)




# try_fgsm('RT')










































def hw3_pgd(dataset, sess, epochs, epsilon):
    # TODO: Implement PGD and return X_adv, the adversarial examples produced.
    # TODO: Evaluate the overall accuracy of the of the model on the adversarial example in each epoch.

    # target_model, X_seed, y_seed, epsilon, epoch, sess
    data = pickle.load( open( "datasets/{}/{}_embeddings.p".format(dataset, dataset), "rb" ) )


    # model = pickle.load( open( "models/CNN/CNN_SA_{}.p".format(dataset), "rb" ) )
    model = tf.keras.models.load_model('models/CNN/CNN_SA_{}.h5'.format(dataset))

    x_test_pos = data['test']['pos'][0:5]
    x_test_neg = data['test']['neg'][0:5]
    x_test = []
    x_test.extend(x_test_pos)
    x_test.extend(x_test_neg)
    y_test = [1 for i in range(len(x_test_pos))]
    y_test.extend([0 for i in range(len(x_test_neg))])
    x_test = np.array(x_test, dtype='float')
    y_test = np.array(y_test, dtype='float')


    X_seed = np.copy(x_test)
    y_seed = np.copy(y_test)

    y_s = tf.placeholder('float', None) # plug in y_seed
    model_input = model.input
    model_output = model.output
    loss = tf.keras.backend.sparse_categorical_crossentropy(y_s, model_output)
    gradient_tensor = tf.gradients(loss, model_input)

    X_adv_N = np.copy(X_seed)
    y_seed_N = np.copy(y_seed)

    for i in range(epochs):
        print(i)
        print(X_adv_N)
        print(y_seed_N)
        gradient = sess.run(
            gradient_tensor, 
            feed_dict = {
                model_input: X_adv_N,
                y_s: y_seed_N
            }
        )
        print(gradient)
        # X_adv_N = np.clip(X_adv_N + epsilon * np.sign(gradient), 0, 1)[0]
        X_adv_N = X_adv_N + epsilon * np.sign(gradient)
        print(X_adv_N)
        # overall_evaluate(target_model, X_adv_N, y_seed_N)
    return X_adv_N


# with tf.Session() as sess:
#     hw3_pgd('RT',sess, 3, 0.01)



def sample_grad(dataset):

    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    outputTensor = model.output #Or model.layers[index].output
    listOfVariableTensors = model.trainable_weights
    gradients = k.gradients(outputTensor, listOfVariableTensors)
    trainingExample = np.random.random((1,8))

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    evaluated_gradients = sess.run(gradients,feed_dict={model.input:trainingExample})
    # print(len(evaluated_gradients))
    # for single in evaluated_gradients:
    #     pprint.pprint(single)

    ce_loss = k.mean(k.categorical_crossentropy(y_true, model.output))
    gradients = k.gradients(ce_loss,model.inputs)
    trainingExample = np.random.randint(50,size=(1,20))

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    evaluated_gradients = sess.run(gradients, feed_dict={model.input:trainingExample})





    y_true = k.reshape(y_true, (-1, num_classes))
    y_pred = k.reshape(y_pred, (-1, num_classes))

    # an input layer to feed labels
    y_true = Input(shape=labels_shape)
    # compute loss based on model's output and true labels
    ce = K.mean(K.categorical_crossentropy(y_true, model.output))
    # compute gradient of loss with respect to inputs
    grad_ce = K.gradients(ce, model.inputs)
    # create a function to be able to run this computation graph
    func = K.function(model.inputs + [y_true], grad_ce)

    # usage
    output = func([model_input_array(s), true_labels])


# sample_grad('RT')


















def get_map_index_to_sentence(document):
    sentences = get_sentences(document)
    map_index_to_sentence = {}
    for i in range(len(sentences)):
        map_index_to_sentence[i] = sentences[i]




def get_sentences(tokens):
    original_review = TreebankWordDetokenizer().detokenize(tokens)
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer')) # updated
    doc = nlp(original_review)
    sentences = [sent.string.strip() for sent in doc.sents]
    return sentences



# get_sentences(['I','really','like','pie','.','How','about','you','?','I','had','a','bad','day','today','.'])






def rank_sentences(sentences, whole_document_score, classifier=None):

    map_sentence_to_importance = {}
    # for i in range(len(sentences)):
    for i in range(len(sentences)):
        classifier_score = random.random()  ## Change later
        if whole_document_score == 0:
            importance = 0.5 - classifier_score
        else:
            importance = classifier_score - 0.5

        if (importance > 0):
            map_sentence_to_importance[i] = importance

    sentences_sorted_by_importance = {k: v for k, v in sorted(map_sentence_to_importance.items(), key=lambda item: -item[1])}
    print(sentences_sorted_by_importance)
    return sentences_sorted_by_importance





def get_importances_of_words_in_sentence(sentence, classifier_score, classifier):
    sentence_tokens = nltk.word_tokenize(sentence)

    word_importances = {}
    for curr_token in sentence_tokens:
        sentence_tokens_without = [token for token in sentence_tokens if token != curr_token]
        sentence_without = TreebankWordDetokenizer().detokenize(sentence_tokens_without)
        word_score = random.random()
        if (classifier_score == 0):
            word_importance = 1 - word_score
        else:
            word_importance = word_score

        word_importances[curr_token] = word_importance
    word_importances = {k: v for k, v in sorted(word_importances.items(), key=lambda item: -item[1])}
    print(word_importances)

    return word_importances




# sentences = get_sentences(['I','really','like','pie','like','you','.','How','about','you','?','I','had','a','bad','day','today','.'])
# rank_sentences(sentences,1)


# get_importances_of_words_in_sentence(sentences[0], 1, None)



def strops():
    str1 = "english language neutral " 
    str2 = "I like to walk."

    res = str1+str2
    print(res)
# strops()



def get_inverse_OHE(row):

    if row['toxic']==1 or row['severe_toxic']==1 or row['obscene']==1 or row['threat']==1  or row['insult']==1 or row['identity_hate']==1:
        return 1
    return 0




def get_train_kaggle():
    df = pd.read_csv("datasets/Kaggle/raw/train.csv")

    df['score'] = df.apply (lambda row: get_inverse_OHE(row), axis=1)
    df = df[['comment_text','score']]

    data = {'train':{'pos':[], 'neg':[]}, 'test':{'pos':[], 'neg':[]}}
    for index, row in df.iterrows():
        if (row['score'] == 1):
            data['train']['pos'].append(row['comment_text'])
        else:
            data['train']['neg'].append(row['comment_text'])

    pickle.dump(data, open( "datasets/Kaggle/Kaggle_train_tokens.p", "wb" ))

# get_train_kaggle()

def get_test_token_kaggle():
    df = pd.read_csv("datasets/Kaggle/raw/test.csv")
    df2 = pd.read_csv('datasets/Kaggle/raw/test_labels.csv')

    df = pd.concat([df, df2], axis=1)
    df=df.drop(['id'],axis=1)
    df = df[df.toxic != -1]

    df['score'] = df.apply (lambda row: get_inverse_OHE(row), axis=1)
    df = df[['comment_text','score']]

    data = pickle.load( open( "datasets/Kaggle/Kaggle_train_tokens.p", "rb" ) )
    for index, row in df.iterrows():
        if (row['score'] == 1):
            data['test']['pos'].append(row['comment_text'])
        else:
            data['test']['neg'].append(row['comment_text'])

    pickle.dump(data, open( "datasets/Kaggle/Kaggle_sentences.p", "wb" ))


# get_test_token_kaggle()





def get_tokens_kaggle():
    data = pickle.load( open( "datasets/Kaggle/Kaggle_sentences.p", "rb" ) )

    res = {'train':{'pos':[], 'neg':[]}, 'test':{'pos':[], 'neg':[]}}


    for key1 in data:
        for key2 in data[key1]:
            for sentence in data[key1][key2]:
                res[key1][key2].append(nltk.word_tokenize(sentence))
    pickle.dump(res, open( "datasets/Kaggle/Kaggle_tokens.p", "wb" ))

# get_tokens_kaggle()





def get_vectors_kaggle():


    start = time.time()
    with open('old/glove_vectors.json') as f:
        glove_vectors = json.load(f)
    end = time.time()
    print("DONE LOADING: {} minutes".format((end-start)/60))




    data = pickle.load( open( "datasets/Kaggle/Kaggle_tokens.p", "rb" ) )
    res = {'train':{'pos':[], 'neg':[]}, 'test':{'pos':[], 'neg':[]}}

    for key1 in data:
        for key2 in data[key1]:
            print("{} | {} | {}".format(key1, key2, len(data[key1][key2])))

    for key1 in data:
        for key2 in data[key1]:
            for token_list in data[key1][key2]:
                vector = transform_to_feature_vector(token_list, glove_vectors)
                res[key1][key2].append(vector)

    for key1 in res:
        for key2 in res[key1]:
            print("{} | {} | {}".format(key1, key2, len(res[key1][key2])))

    pickle.dump(res, open( "datasets/Kaggle/Kaggle_vectors.p", "wb" ))



# get_vectors_kaggle()





def get_embeds_kaggle():
    data = pickle.load( open( "datasets/Kaggle/Kaggle_tokens.p", "rb" ) )

    words = {}
    for key1 in data:
        for key2 in data[key1]:
            for tokens_list in data[key1][key2]:
                for word in tokens_list:
                    words[word] = 0

    for key1 in data:
        for key2 in data[key1]:
            for tokens_list in data[key1][key2]:
                for word in tokens_list:
                    words[word] += 1

    words = {k: v for k, v in sorted(words.items(), key=lambda item: -item[1])}

    idx = 0
    for word in words:
        words[word] = idx
        idx += 1

    idx2= 0 
    for word in words:
        print("{} | {}".format(word, words[word]))
        idx2 +=1 
        if idx2 > 10:
            break

    pickle.dump(words, open( "datasets/Kaggle/Kaggle_embed_map.p", "wb" ))

# get_embeds_kaggle()



def get_glove_kaggle():
    start = time.time()
    # with open('old/glove_vectors.json') as f:
    with open('old/glove_vectors.json') as f:
        glove_vectors = json.load(f)
    end = time.time()
    print("DONE LOADING: {} minutes".format((end-start)/60))



    res = {}
    num_in = 0
    num_out = 0
    data = pickle.load( open( "datasets/Kaggle/Kaggle_tokens.p", "rb" ) )
    for key1 in data:
        for key2 in data[key1]:
            for token_list in data[key1][key2]:
                for word in token_list:
                    if (word in glove_vectors):
                        res[word] = glove_vectors[word]
                        num_in += 1
                    else:
                        res[word] = [(random.random()/5)-0.1 for i in range(300)]
                        num_out += 1

    with open('datasets/Kaggle/glove_kaggle.json', 'w') as outfile:
        json.dump(res, outfile)

# get_glove_kaggle()


def get_len_standard_tcc(review, max_len, embed_map):
    res = []

    for token in review:
        res.append(embed_map[token])
    
    if (len(res) > max_len):
        res = res[0:max_len]
    elif (len(res) < max_len):
        diff = max_len - len(res)
        for i in range(0,diff):
            res.append(random.randint(1,10))

    return res


def get_kaggle_embeds_same_len():
    data = pickle.load( open( "datasets/Kaggle/Kaggle_tokens.p", "rb" ) )
    embed_map = pickle.load( open( "datasets/Kaggle/Kaggle_embed_map.p", "rb" ) )

    res = {'train':{'pos':[], 'neg':[]}, 'test':{'pos':[], 'neg':[]}}

    for key1 in data:
        for key2 in data[key1]:
            for review in data[key1][key2]:
                embed = get_len_standard_tcc(review, 40, embed_map)
                res[key1][key2].append(embed)

    for key1 in res:
        for key2 in res[key1]:
            print(len(random.choice(res[key1][key2])))
            break
        break

    pickle.dump(res, open( "datasets/Kaggle/Kaggle_embeddings.p", "wb" ))




# get_kaggle_embeds_same_len()



def get_full_embed_map():   
    word_to_index = pickle.load( open( "datasets/Kaggle/Kaggle_embed_map.p", "rb" ) )
    index_to_word = {}
    for k,v in word_to_index.items():
        index_to_word[v] = k
    
    res = {}
    res['w2i'] = word_to_index
    res['i2w'] = index_to_word

    pickle.dump(res, open( "datasets/Kaggle/Kaggle_embed_map_full.p", "wb" ))

    print(len(res['w2i']))
    print(len(res['i2w']))



get_full_embed_map()