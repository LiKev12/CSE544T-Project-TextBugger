
from operator import add
import numpy as np

import json

def get_glove_IMDB():
    with open('glove_vectors.json') as json_file:
        glove_vectors = json.load(json_file)

    imdb_path = "data/IMDB_text_tokenized.p"
    imdb_tokens = pickle.load(imdb_path)


    imdb_glove = {}
    for key1 in imdb_tokens:
        for key2 in imdb_tokens:
            sentence = imdb_tokens[key1][key2]
            for word in sentence:
                if word in glove_vectors:
                    imdb_glove[word] = glove_vectors[word]

    with open('Glove_IMDB.json', 'w') as f:
        json.dump(imdb_glove, f)



    imdb_path = "data/IMDB_text_tokenized.p"
    imdb_tokens = pickle.load(imdb_path)
    
    rt_glove = {}
    for key1 in rt_tokens:




get_glove_IMDB()




