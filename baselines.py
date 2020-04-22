from __future__ import absolute_import, division, print_function, unicode_literals

import json
import pickle
import random
import statistics
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import keras
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten, MaxPooling1D, Flatten, Conv2D, MaxPooling2D
import keras.backend as K

from scipy.optimize import fmin_tnc
from sklearn.neighbors import NearestNeighbors

from textbugger_utils import *



class Baseline():

    def __init__(self, F, attack_type, model_type, glove_vectors, embed_map, dataset, max_len, num_epochs):
        self.min_ = 0
        self.max_ = 1
        self.F = F # Classifier/Model
        self.attack_type = attack_type
        self.model_type = model_type
        self.glove_vectors = glove_vectors
        self.embed_map = embed_map
        self.dataset = dataset
        self.max_len = max_len
        self.num_epochs = num_epochs

        self.glove_embeddings = np.array(list(glove_vectors.values()))
        self.glove_words = np.array(list(glove_vectors.keys()))



    def num_encode(self, column):
        # binary encode
        # enc = OneHotEncoder(sparse=False)
        enc = OneHotEncoder()
        column = column.reshape(-1, 1)
        enc.fit(column)
        encode_col = enc.transform(column).toarray()
        return encode_col


    def load_glove(self):
        glove_vectors = json.load(open("datasets/glove_final.json", "rb"))
        # extend glove vector
        DIM = 300
        symbols = {'<pad>': [1e-8] * DIM, '<bos>': [1] * DIM, '<eos>': [2] * DIM}
        glove_vectors.update(symbols)
        glove_embeddings = list(glove_vectors.values())
        glove_words = np.array(list(glove_vectors.keys()))
        return glove_vectors,glove_embeddings,glove_words


    # load dataset glove embedding vectors
    def load_3D_data(self):
        tokens = pickle.load(open("datasets/{}/{}_tokens.p".format(self.dataset,self.dataset), "rb"))
        print("open tokens file")

        # tokens x,y
        # train
        x_train_pos_tokens = tokens['train']['pos']
        x_train_neg_tokens = tokens['train']['neg']
        x_train_tokens = []
        x_train_tokens.extend(x_train_pos_tokens)
        x_train_tokens.extend(x_train_neg_tokens)
        # test
        x_test_pos_tokens = tokens['test']['pos']
        x_test_neg_tokens = tokens['test']['neg']
        x_test_tokens = []
        x_test_tokens.extend(x_test_pos_tokens)
        x_test_tokens.extend(x_test_neg_tokens)

        x_train_tokens = np.array(x_train_tokens)
        x_test_tokens = np.array(x_test_tokens)
        self.x_train_tokens = x_train_tokens
        self.x_test_tokens = x_test_tokens
        #print("111111111")

        res = {}
        for key1 in tokens:
            res[key1] = {}
            for key2 in tokens[key1]:
                res[key1][key2] = []

        for key1 in tokens:
            for key2 in tokens[key1]:
                vects = []
                for doc in tokens[key1][key2]:
                    #print('2222222')
                    vect = transform_to_word_feature_vector(doc, self.glove_vectors, self.max_len)
                    #print("4444444")
                    vects.append(vect)
                res[key1][key2] = vects
        #print("5555555")
        self.dataset_glove_embeddings = res
        pickle.dump(res,open("datasets/{}/{}_{}_glove_vectors.p".format(self.dataset, self.model_type, self.dataset),"wb"))
        print("write glove file")

        # glove word embeddings x,y
        # train
        x_train_pos = res['train']['pos']
        x_train_neg = res['train']['neg']
        x_train_glove = []
        x_train_glove.extend(x_train_pos)
        x_train_glove.extend(x_train_neg)
        # test
        x_test_pos = res['test']['pos']
        x_test_neg = res['test']['neg']
        x_test_glove = []
        x_test_glove.extend(x_test_pos)
        x_test_glove.extend(x_test_neg)

        x_train_glove = np.array(x_train_glove)
        x_test_glove = np.array(x_test_glove)
        self.x_train_glove = x_train_glove
        self.x_test_glove = x_test_glove


        # glove mean embeddings x,y
        glove_mean = pickle.load(open("datasets/{}/{}_vectors.p".format(self.dataset, self.dataset), "rb"))
        print("open glove mean file")
        glove_mean = dict(glove_mean)
        self.dataset_glove_mean_embeddings = glove_mean

        ## Train
        x_train_pos_mean = glove_mean['train']['pos']
        x_train_neg_mean = glove_mean['train']['neg']
        x_train_glove_mean = []
        x_train_glove_mean.extend(x_train_pos_mean)
        x_train_glove_mean.extend(x_train_neg_mean)
        y_train = [1 for i in range(len(x_train_pos_mean))]
        y_train.extend([0 for i in range(len(x_train_neg_mean))])
        x_train_glove_mean = np.array(x_train_glove_mean)
        y_train = np.array(y_train)

        ## Test
        x_test_pos_mean = glove_mean['test']['pos']
        x_test_neg_mean = glove_mean['test']['neg']
        x_test_glove_mean = []
        x_test_glove_mean.extend(x_test_pos_mean)
        x_test_glove_mean.extend(x_test_neg_mean)
        y_test = [1 for i in range(len(x_test_pos_mean))]
        y_test.extend([0 for i in range(len(x_test_neg_mean))])
        x_test_glove_mean = np.array(x_test_glove_mean)
        y_test = np.array(y_test)

        self.x_train_glove_mean = x_train_glove_mean
        self.x_test_glove_mean = x_test_glove_mean
        self.y_train = y_train
        self.y_test = y_test

        # y onehot encoding
        y_train_onehot = self.num_encode(np.array(self.y_train))
        y_test_onehot = self.num_encode(np.array(self.y_test))
        self.y_train_onehot = y_train_onehot
        self.y_test_onehot = y_test_onehot


    # Kevin used for Model training
    def load_2D_data(self):
        data = pickle.load(open("datasets/{}/{}_vectors.p".format(self.dataset, self.dataset), "rb"))

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

        # self.x_train = x_train
        # self.x_test = x_test
        # self.y_train = y_train
        # self.y_test = y_test

    def test_dimensions(self):
        print("shape x_train_tokens",self.x_train_tokens.shape)
        print("shape x_test_tokens", self.x_test_tokens.shape)
        print("shape x_train_glove", self.x_train_glove.shape)
        print("shape x_test_glove", self.x_test_glove.shape)
        print("shape x_train_glove_mean", self.x_train_glove_mean.shape)
        print("shape x_test_glove_mean", self.x_test_glove_mean.shape)
        print("shape y_train", self.y_train.shape)
        print("shape y_test", self.y_test.shape)
        print("shape y_train_onehot", self.y_train_onehot.shape)
        print("shape y_test_onehot", self.y_test_onehot.shape)
        print("shape glove embeddings", self.glove_embeddings.shape)
        print("shape glove words", self.glove_words.shape)


    #  ------------------------ Models ---------------------------
    def train_model_get_gradients(self):
        if self.model_type=="LR":
            theta = np.zeros((self.x_train_glove_mean.shape[1], 1))
            self.parameters = self.fit(self.x_train_glove_mean, self.y_train, theta)
            # test to get gradient
            print(self.accuracy(self.x_test_glove_mean, self.y_test.flatten()))
            theta = self.parameters[:, np.newaxis]
            self.gradient(theta, self.x_test_glove_mean, self.y_test)

        elif self.model_type=="LSTM":
            self.make_LSTM()

        elif self.model_type=="CNN":
            self.make_CNN()


    # Logistic Regression Model
    def sigmoid(self, x):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))

    def net_input(self, theta, x):
        # Computes the weighted sum of inputs
        return np.dot(x, theta)

    def probability(self, theta, x):
        # Returns the probability after passing through sigmoid
        return self.sigmoid(self.net_input(theta, x))

    def cost_function(self, theta, x, y):
        # Computes the cost function for all the training samples
        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(
            y * np.log(self.probability(theta, x)) + (1 - y) * np.log(1 - self.probability(theta, x)))
        return total_cost

    def gradient(self, theta, x, y):
        # Computes the gradient of the cost function at the point theta
        m = x.shape[0]
        gradients = (1 / m) * np.dot(x.T, self.sigmoid(self.net_input(theta,   x)) - y)
        self.gradients = gradients
        return gradients

    def fit(self, x, y, theta):
        opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient,args=(x, y.flatten()))
        return opt_weights[0]

    def predict(self, x):
        theta = self.parameters[:, np.newaxis]
        return self.probability(theta, x)

    def accuracy(self, x, actual_classes, probab_threshold=0.5):
        predicted_classes = (self.predict(x) >= probab_threshold).astype(int)
        predicted_classes = predicted_classes.flatten()
        accuracy = np.mean(predicted_classes == actual_classes)
        return accuracy * 100


    # LSTM Model
    def make_LSTM(self):
        data = pickle.load(open("datasets/{}/{}_embeddings.p".format(self.dataset, self.dataset), "rb"))

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

        self.hidden_size = 32
        vocab_size = 170000

        model = Sequential()
        # 'embedding_16/embeddings:0' shape=(170000, 32)
        model.add(Embedding(vocab_size, self.hidden_size))
        # 'lstm_16/kernel:0' shape=(32, 128); 'lstm_16/recurrent_kernel:0' shape=(32, 128); 'lstm_16/bias:0' shape=(128,)
        model.add(LSTM(self.hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2))
        # 'dense_16/kernel:0' shape=(32, 1); 'dense_16/bias:0' shape=(1,)
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Get gradient tensors
        weights = model.trainable_weights  # weight tensors
        weights = [weight for weight in weights if model.get_layer(
            weight.name.split('/')[0]).trainable]  # filter down weights tensors to only ones which are trainable
        # gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
        gradients = model.optimizer.get_gradients(model.total_loss, weights)

        print(weights)

        # Define keras function to return gradients
        input_tensors = [model.inputs[0],  # input data
                         model.sample_weights[0],  # how much to weight each sample by
                         model.targets[0],  # labels
                         K.learning_phase(),  # train or test mode
                         ]
        get_gradients = K.function(inputs=input_tensors, outputs=gradients)

        # Get gradients of weights for particular (X, sample_weight, y, learning_mode) tuple
        inputs = [x_test,  # X
                  np.ones(x_test.shape[0]),  # sample weights
                  y_test,  # y
                  0  # learning phase in TEST mode
                  ]
        # print(weights, get_gradients(inputs))
        self.gradients = get_gradients(inputs)[-2]

        model.fit(x_train, y_train, epochs=self.num_epochs, shuffle=True)
        # use newly trained model
        self.F = model

        loss, acc = model.evaluate(x_test, y_test)
        print(loss, acc)


    # CNN Model
    def make_CNN(self):
        dataset = 'RT'
        data = pickle.load(open("datasets/{}/{}_embeddings.p".format(dataset, dataset), "rb"))
        emb_map = pickle.load(open("datasets/embed_map.p", "rb"))
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

        ## Model
        max_len = x_train.shape[1]
        batch_size = 32
        embedding_dims = 10
        filters = 16
        kernel_size = 3
        self.hidden_size = 250

        model = Sequential()
        # 'embedding_17/embeddings:0' shape=(170000, 10)
        model.add(Embedding(vocab_size, embedding_dims, input_length=max_len))
        model.add(Dropout(0.5))
        # 'conv1d_1/kernel:0' shape=(3, 10, 16) ; 'conv1d_1/bias:0' shape=(16,)
        model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu'))
        model.add(MaxPooling1D())
        # 'conv1d_2/kernel:0' shape=(3, 16, 16); 'conv1d_2/bias:0' shape=(16,)
        model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        # 'dense_17/kernel:0' shape=(48, 250); 'dense_17/bias:0' shape=(250,)
        model.add(Dense(self.hidden_size, activation='relu'))  # (48,250)
        model.add(Dropout(0.5))  # (250,)
        # 'dense_18/kernel:0' shape=(250, 1); 'dense_18/bias:0' shape=(1,)
        model.add(Dense(1, activation='sigmoid'))  # (250,1) (1,)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Get gradient tensors
        weights = model.trainable_weights  # weight tensors
        weights = [weight for weight in weights if model.get_layer(
            weight.name.split('/')[0]).trainable]  # filter down weights tensors to only ones which are trainable
        # gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
        gradients = model.optimizer.get_gradients(model.total_loss, weights)

        print(weights)

        # Define keras function to return gradients
        input_tensors = [model.inputs[0],  # input data
                         model.sample_weights[0],  # how much to weight each sample by
                         model.targets[0],  # labels
                         K.learning_phase(),  # train or test mode
                         ]
        get_gradients = K.function(inputs=input_tensors, outputs=gradients)

        # Get gradients of weights for particular (X, sample_weight, y, learning_mode) tuple
        inputs = [x_test,  # X
                  np.ones(x_test.shape[0]),  # sample weights
                  y_test,  # y
                  0  # learning phase in TEST mode
                  ]
        # print(weights, get_gradients(inputs))
        self.gradients = get_gradients(inputs)[-2]
        #print(np.array(self.gradients).shape)

        model.fit(x_train, y_train, batch_size=batch_size, epochs=self.num_epochs, validation_data=(x_test, y_test))

        loss, acc = model.evaluate(x_test, y_test)
        print('CNN Model -- ACC {} -- LOSS {}'.format(acc, loss))
        print('{} model done!'.format(dataset))




    # Baseline Attacks
    # FGSM
    def fgsm_gradient(self):
        epsilon = 0.01
        if self.model_type=="LR":
            adv_test_embedding = np.swapaxes(self.x_test_glove, 0, 1) + epsilon * np.sign(self.gradients.T)
            adv_test_embedding = np.swapaxes(adv_test_embedding, 0, 1)
        elif self.model_type=="LSTM" or self.model_type=="CNN":
            gradients = self.gradients.reshape(self.hidden_size, 1, 1)
            epsilon = 0.01
            num = int(len(self.x_test_glove) / self.hidden_size)
            adv_test_embedding = []
            for i in range(num):
                x = self.x_test_glove[i * self.hidden_size:(i + 1) * self.hidden_size]
                adv_x = np.add(x, epsilon * np.sign(gradients))
                adv_test_embedding.extend(adv_x)

            remain_length = len(self.x_test_glove) - (i + 1) * self.hidden_size
            x = self.x_test_glove[(i + 1) * self.hidden_size:]
            adv_x = np.add(x, epsilon * np.sign(gradients[:remain_length]))
            adv_test_embedding.extend(adv_x)
            adv_test_embedding = np.array(adv_test_embedding)
            print("all_adv_x shape:",adv_test_embedding.shape)
        self.adv_test_embedding = adv_test_embedding

    # DeepFool
    def deepfool_gradient(self):
        adv_test_embedding = []
        for doc in self.x_test_tokens:
            i = 0
            r = []
            x = []
            x.append(doc)
            y0 = get_prediction_given_tokens(self.model_type, self.F, x[0], self.glove_vectors, self.embed_map, self.dataset)
            yi = get_prediction_given_tokens(self.model_type, self.F, x[i], self.glove_vectors, self.embed_map, self.dataset)
            while np.sign(y0)==np.sign(yi):
                new_r = - (yi/np.linalg.norm(self.gradients.T[i]))*self.gradients.T[i]
                print("gradient shape",self.gradients.shape)  # (300,5332) [i] (300)
                print("yi",yi)
                print("norm",np.linalg.norm(self.gradients[i]))
                print("new r shape",new_r.shape)
                r.append(new_r)
                print("xi shape", np.array(x[i]).shape)

                x.append(x[i]+r[i])
                i += 1
                yi = get_prediction_given_tokens(self.model_type, self.model, x[i], self.glove_vectors, self.embed_map,
                                                 self.dataset)
            adv_test_embedding.append(np.mean(r))
        adv_test_embedding = np.array(adv_test_embedding)
        self.adv_test_embedding = adv_test_embedding





    # Calculate Nearest Neighbor in Glove Embedding
    def compute_nns(self):
        adv_tokens = {}
        print("start compute nns")
        for i in range(40):
            rand_idx = random.randint(1, self.adv_test_embedding.shape[0])
            print("i=", i, "rand idx=", rand_idx)
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.glove_embeddings)
            distances, indices = nbrs.kneighbors(self.adv_test_embedding[rand_idx])
            adv_tokens[rand_idx] = self.glove_words[indices].T.tolist()[0]
            print(adv_tokens)
        pickle.dump(adv_tokens, open("attacks/{}_{}_{}_adv_tokens.p".format(self.model_type,self.dataset,self.attack_type), "wb"))
        adv_tokens = np.array(adv_tokens)
        self.test_adv_tokens = adv_tokens


    # Evaluate Attack
    def computePerturbedWord(self):
        token_idx = list(self.test_adv_tokens.keys())
        original_tokens = self.x_test_tokens[token_idx]
        # for i in range(len(token_idx)):
        #     print("len ori",len(original_tokens.tolist()[i]),"len adv",len(list(self.test_adv_tokens.values())[i]))
        flatten_ori = [item for sublist in original_tokens.tolist() for item in sublist[:20]]
        flatten_adv = [item for sublist in list(self.test_adv_tokens.values()) for item in sublist]
        # print(len(flatten_ori),len(flatten_adv))
        num_perturb = 0
        for i in range(len(flatten_ori)):
            if(flatten_ori[i] != flatten_adv[i]):
                num_perturb += 1
        perturb_rate = num_perturb/len(flatten_ori)
        return perturb_rate


    def computeSuccessRate(self):
        num_success = 0
        for idx in list(self.test_adv_tokens.keys()):
            doc = self.test_adv_tokens[idx]
            y_pred = get_prediction_given_tokens(self.model_type, self.F, doc, self.glove_vectors, self.embed_map, self.dataset)
            if self.y_test[idx] != y_pred:
                num_success += 1
        success_rate = num_success/(len(list(self.test_adv_tokens.keys())))
        return success_rate








if __name__=="__main__":
# def run(dataset, model_type, attack_type, num_epochs):
    dataset = 'IMDB'
    model_type = 'LSTM'
    hidden_size = 32
    attack_type = 'fgsm'
    num_epochs = 3
    print("dataset=",dataset,"model=",model_type,"attack=",attack_type)
    if (model_type == 'LR'):
        if (dataset == 'IMDB'):
            model = pickle.load( open( "models/LR/LR_SA_IMDB.p", "rb" ))
        elif(dataset == 'RT'):
            model = pickle.load( open( "models/LR/LR_SA_RT.p", "rb" ))
        elif (dataset == 'Kaggle'):
            model = pickle.load(open("models/LR/LR_TCD_Kaggle.p", "rb"))

    elif (model_type == 'LSTM'):
        if (dataset == 'IMDB'):
            model = pickle.load( open( "models/LSTM/LSTM_SA_IMDB.p", "rb" ))
        elif(dataset == 'RT'):
            model = pickle.load( open( "models/LSTM/LSTM_SA_RT.p", "rb" ))
        elif (dataset == 'Kaggle'):
            model = pickle.load(open("models/LSTM/LSTM_TCD_Kaggle.p", "rb"))

    elif (model_type == 'CNN'):
        if (dataset == 'IMDB'):
            model = pickle.load( open( "models/CNN/CNN_SA_IMDB.p", "rb" ))
        elif(dataset == 'RT'):
            model = pickle.load( open( "models/CNN/CNN_SA_RT.p", "rb" ))
        elif(dataset == 'Kaggle'):
            model = pickle.load(open("models/CNN/CNN_TCD_Kaggle.p", "rb"))
    print("load model")

    glove_vectors = json.load(open("datasets/glove_final.json", "rb"))
    print("load glove vectors")

    if(dataset=='IMDB' or dataset=='RT'):
        embed_map = pickle.load(open("datasets/embed_map.p", "rb"))
    else:
        embed_map = pickle.load(open("datasets/Kaggle/Kaggle_embed_map.p", "rb"))
    print("load embed_map")

    if (dataset == 'RT'):
        max_len = 20
    else:
        max_len = 200
    print("set max_len")

    myBaseline = Baseline(model, attack_type, model_type, glove_vectors, embed_map, dataset, max_len, num_epochs)
    print("object created")
    # load data
    #glove_vectors,glove_embeddings,glove_words = myBaseline.load_glove()
    #myBaseline.load_2D_data()
    myBaseline.load_3D_data()
    myBaseline.test_dimensions()
    print("load data done")
    #
    # train model
    # myBaseline.train_model_get_gradients()
    # pickle.dump(myBaseline.gradients,open("models/{}/{}_gradient.p".format(myBaseline.model_type,myBaseline.dataset),"wb"))
    myBaseline.hidden_size = hidden_size
    myBaseline.gradients = pickle.load( open( "models/{}/{}_gradient.p".format(myBaseline.model_type,myBaseline.dataset), "rb" ))
    print("train LR model and get gradient done")

    # baseline adversarial attack
    if myBaseline.attack_type=='fgsm':
        myBaseline.fgsm_gradient()
    else:
        myBaseline.deepfool_gradient()
    pickle.dump(myBaseline.adv_test_embedding, open("attacks/{}_{}_{}_adv_embeddings.p".format(myBaseline.model_type, myBaseline.dataset, myBaseline.attack_type), "wb"))

    print("adv embedding shape",myBaseline.adv_test_embedding.shape)
    print("done generate adversarial examples")
    myBaseline.compute_nns()

    # evaluate attack
    test_adv_tokens = dict(pickle.load(open("attacks/{}_{}_{}_adv_tokens.p".format(myBaseline.model_type,myBaseline.dataset,myBaseline.attack_type), "rb")))
    myBaseline.test_adv_tokens = test_adv_tokens
    perturb_rate = myBaseline.computePerturbedWord()
    print("perturb rate is", perturb_rate)
    success_rate = myBaseline.computeSuccessRate()
    print("success rate is", success_rate)