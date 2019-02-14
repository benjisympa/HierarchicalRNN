import glob
import json
import re
import pickle
import pandas as pd
import spacy
import string
import numpy as np
import itertools
import csv
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split

import torchtext.vocab as vocab
nlp = spacy.load('en')

def save(nameFile, toSave):
    pickle_out = open(nameFile+".pickle", "wb")
    pickle.dump(toSave, pickle_out)
    pickle_out.close()

def load(nameFile):
    pickle_in = open(nameFile+".pickle", "rb")
    return pickle.load(pickle_in)

def get_word_vector(word):
    return we.vectors[we.stoi[word]]

def closest(vec, n=2):#10):
    """
    Find the closest words for a given vector
    """
    all_dists = [(w, torch.dist(vec, get_word_vector(w))) for w in we.itos]
    return sorted(all_dists, key=lambda t: t[1])[:n]

def print_tuples(tuples):
    for tuple in tuples:
        print('(%.4f) %s' % (tuple[1], tuple[0]))

# In the form w1 : w2 :: w3 : ?
def analogy(w1, w2, w3, n=5, filter_given=True):
    print('\n[%s : %s :: %s : ?]' % (w1, w2, w3))
   
    # w2 - w1 + w3 = w4
    closest_words = closest(get_word_vector(w2) - get_word_vector(w1) + get_word_vector(w3))
    
    # Optionally filter out given words
    if filter_given:
        closest_words = [t for t in closest_words if t[0] not in [w1, w2, w3]]

    print_tuples(closest_words[:n])

def load_data(path_transcripts='/vol/work2/galmant/transcripts/'):
    punctuations_end_sentence = ['.', '?', '!']

    we = vocab.FastText(language='en')
    '''
    pretrained_aliases = {
        "charngram.100d": partial(CharNGram),
        "fasttext.en.300d": partial(FastText, language="en"),
        "fasttext.simple.300d": partial(FastText, language="simple"),
        "glove.42B.300d": partial(GloVe, name="42B", dim="300"),
        "glove.840B.300d": partial(GloVe, name="840B", dim="300"),
        "glove.twitter.27B.25d": partial(GloVe, name="twitter.27B", dim="25"),
        "glove.twitter.27B.50d": partial(GloVe, name="twitter.27B", dim="50"),
        "glove.twitter.27B.100d": partial(GloVe, name="twitter.27B", dim="100"),
        "glove.twitter.27B.200d": partial(GloVe, name="twitter.27B", dim="200"),
        "glove.6B.50d": partial(GloVe, name="6B", dim="50"),
        "glove.6B.100d": partial(GloVe, name="6B", dim="100"),
        "glove.6B.200d": partial(GloVe, name="6B", dim="200"),
        "glove.6B.300d": partial(GloVe, name="6B", dim="300")
    }
    '''

    X_all = []
    Y_all = []
    words_set = set()
    for f in sorted(glob.glob(path_transcripts+'*')):
        with open(f, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            X_ = []
            Y_ = []
            for row in reader:
                sentence = row[2]
                old_word = row[2]
                for word in row[3:]:
                    if any(punctuation in old_word for punctuation in punctuations_end_sentence) and word and word[0].isupper():
                        X_.append(sentence)
                        Y_.append(row[1])
                        sentence = word
                    else:
                        sentence += ' '+word
                    old_word = word
                X_.append(sentence)
                Y_.append(row[1])
            X = [s.lower().split() for s in X_]
            Y = [s.lower() for s in Y_]
            to_del = []
            for s in X:
                for w in s:
                    if w not in we.stoi:
                        to_del.append(w)
            X = [[w for w in s if w not in to_del] for s in X]
            for s in X:
                words_set = words_set.union(set(s))
            X_all.append(X)
            Y_all.append(Y)
            assert len(X) == len(Y)

    threshold_train_dev = int(len(X_all)*0.8)
    threshold_dev_test = threshold_train_dev + int(len(X_all)*0.1)
    X_train = X_all[:threshold_train_dev]
    Y_train = Y_all[:threshold_train_dev]
    X_dev = X_all[threshold_train_dev:threshold_dev_test]
    Y_dev = Y_all[threshold_train_dev:threshold_dev_test]
    X_test = X_all[threshold_dev_test:]
    Y_test = Y_all[threshold_dev_test:]
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test, words_set, we