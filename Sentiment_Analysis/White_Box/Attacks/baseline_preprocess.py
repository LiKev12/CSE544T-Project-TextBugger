import re

def preprocess(input_text):
    sentence = input_text.split(".")

    for i in range(0, len(sentence)):
        sentence[i] = re.sub("\-", ' ', sentence[i])
        sentence[i] = re.sub("[^a-zA-Z' ]+", '', sentence[i])

    sentence = ".".join(sentence)
    sentence = re.sub(" +", " ", sentence)
    return sentence