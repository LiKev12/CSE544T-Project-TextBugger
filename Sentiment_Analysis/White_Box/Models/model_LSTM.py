# https://www.youtube.com/watch?v=UxiDUrOhnf4
# https://github.com/laxmimerit/IMDB-Review-Sentiment-Classification-using-RNN-LSTM/blob/master/IMDB_Review_Sentiment_Classification_using_RNN_LSTM.ipynb

import numpy as np
import matplotlib.pyplot as plt
import tensorflow

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(tf.__version__)