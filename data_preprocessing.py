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
nltk.download('punkt')


file_path = "data/glove.840B.300d/"
## Only use once to get JSON
def loadGloveVectors():
    embeddings_dict = {}
    with open(file_path+"glove.840B.300d.txt", 'r', encoding='utf-8') as f:
        print("opened")
        for line in f:
            # if (count > 5):
            #     break
            values = line.split()
            word = values[0]
            #vector = np.asarray(values[1:], "float32")
            vector = np.asarray(values[1:])
            # print(vector)
            vector = vector.tolist()
            embeddings_dict[word] = vector

            # count += 1

    with open('glove_vectors.json', 'w') as fp:
        json.dump(embeddings_dict, fp)


def loadIMDBData():
    IMDB_text = {}
    IMDB_text["train"]={}
    IMDB_text["train"]["pos"] = []
    IMDB_text["train"]["neg"] = []

    IMDB_text["val"] = {}
    IMDB_text["val"]["pos"] = []
    IMDB_text["val"]["neg"] = []

    IMDB_text["test"] = {}
    IMDB_text["test"]["pos"] = []
    IMDB_text["test"]["neg"] = []

    # base_path = "dataset_imdb"
    base_path = "data/IMDB"
    sets = ["train","test"]
    labels = ["pos","neg"]

    for dataset in sets:
        for label in labels:
            i = 0
            src_path = os.path.join(base_path, dataset, label)
            print(src_path)
            files = os.listdir(src_path)
            for file in files:
                if not os.path.isdir(file):
                    f = smart_open.open(os.path.join(src_path, file),'r',encoding='utf-8')
                    tokens = nltk.word_tokenize(f.readlines()[0])
                    IMDB_text[dataset][label].append(tokens)
                    # for i,line in enumerate(f):
                    #     tokens = gensim.utils.simple_preprocess(f.readlines)
                    #     if set=="train":
                    #         # taggedDocument assign an unique id for each document
                    #         IMDB_text[set][label].append(TaggedDocument(tokens,[i]))
                    #     else:
                    #         IMDB_text[set][label].append(tokens)


    for label in labels:
        val_set = []
        train_id, val_id = train_test_split(np.arange(12500), test_size=0.2)
        val_set.extend(list(np.asarray(IMDB_text["train"][label])[val_id]))
        IMDB_text["val"][label]=val_set

    with open('data/IMDB_text_tokenized.p', 'wb') as fp:
        pickle.dump(IMDB_text, fp)
    #np.save('data/IMDB_text', np.asarray(IMDB_text))


def getIMDBDoc2Vec():
    #data = dict(np.load("data/IMDB_text.npy", allow_pickle=True))
    with open('data/IMDB_text_tokenized.p','rb') as fp:
        data = pickle.load(fp)
    IMDB_vectors = {}
    IMDB_vectors["train"] = {}
    IMDB_vectors["train"]["pos"] = []
    IMDB_vectors["train"]["neg"] = []
    IMDB_vectors["val"] = {}
    IMDB_vectors["val"]["pos"] = []
    IMDB_vectors["val"]["neg"] = []
    IMDB_vectors["test"] = {}
    IMDB_vectors["test"]["pos"] = []
    IMDB_vectors["test"]["neg"] = []
    sets = ["train","val","test"]
    labels = ["pos","neg"]
    # train a Doc2Vec model
    model = Doc2Vec(vector_size=300, min_count=2, epochs=40)
    train_data = data["train"]["pos"]
    train_data.extend(data["train"]["neg"])
    model.build_vocab(train_data)
    model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)
    i=0
    for set in sets:
        for label in labels:
            IMDB_vectors[set][label] = model.infer_vector(IMDB_vectors[set][label])
            print(i,IMDB_vectors[set][label])
            if i>5:
                return
            i+=1

def getIMDBGlove():
    # loadGloveVectors()
    # IMDB_glove_vectors = {}
    # with open('glove_vectors.json') as json_file:
    #     glove_vectors = json.load(json_file)
    # IMDB_glove_vectors = glove_vectors[word]
    pass


def loadMRData():
    MR_text = {}
    MR_text["pos"] = []
    MR_text["neg"] = []

    base_path = "data/MR"
    labels = ["pos", "neg"]

    for label in labels:
        i = 0
        file = os.path.join(base_path, 'rt-polarity.'+label)
        f = smart_open.open(file, 'r', encoding='latin-1')
        for line in f.readlines():
            tokens = nltk.word_tokenize(line)
            MR_text[label].append(tokens)

    with open('data/MR_text_tokenized.p', 'wb') as fp:
        pickle.dump(MR_text, fp)

def getMRData():
    with open('data/MR_text_tokenized.p','rb') as fp:
        data = pickle.load(fp)
    print(data["pos"])


if __name__=="__main__":
    # Only use once to get JSON
    #loadGloveVectors()

    # transform tokenized IMDB and MR data into numpy array
    #loadIMDBData()
    loadMRData()

    # read data from pickle
    getMRData()

    # convert data to vectors
    #getIMDBDoc2Vec()

