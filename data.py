import glob
import json
import re
import pickle
import pandas as pd
import spacy
import string
import numpy as np
import itertools
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

def main_iter_files():
    print('Import data')
    output_path = '/people/maurice/ownCloud/outputGentle/'
    wordsTimeds = []
    for file in sorted(glob.glob(output_path + '*')):
        #print(file)
        if 'wordsTimed' in file and 'pickle' not in file:
            wordsTimed = pd.read_csv(file)  # load(file)
            #print(wordsTimed.head())
            wordsTimeds.append(wordsTimed)
            # wordsTimedGby = wordsTimed.groupby('idSentence')
            '''sentenceTimed = wordsTimedGby.apply(lambda x: x.count())
            sentenceTimed[1] = sentenceTimed.astype(np.float)/len(g) 
            print sentenceTimed'''
    return wordsTimeds

class sentenceTimed(object):
    def __init__(self, wt):
        self.wt = wt
        self.reset(0)
        
    def reset(self, i):
        self.speaker = self.wt.iloc[i].speaker
        self.sentence_courante = ''
        if i > 0:
            self.sentence_courante += self.wt.iloc[i].word
        
    def modif_per_word(self, i):
        self.sentence_courante += ' ' + self.wt.iloc[i].word

    def modif_per_sentence(self, df, i):
        self.add_sentence_informations_to_dataframe(df)
        self.reset(i)
        
    def add_sentence_informations_to_dataframe(self, df):
        df.loc[len(df)] = [self.speaker, self.sentence_courante]
        # print(df)

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

def process_one_file(i, wt):
    #print(i)
    sentencesTimed = pd.DataFrame(columns=['speaker', 'sentence_courante'])    

    st = sentenceTimed(wt)

    punctuation_end_sentence = ['!', '.', '?']
    for word in range(len(wt)):
        if word == 0:
            st.modif_per_word(word)
        elif wt.iloc[word].word[0].isupper() and wt.iloc[word - 1].word in punctuation_end_sentence:
            st.modif_per_sentence(sentencesTimed, word)
        else:
            st.modif_per_word(word)

    st.add_sentence_informations_to_dataframe(sentencesTimed)
    return sentencesTimed

def load_data():
    wts = main_iter_files()

    # Lent
    #punctuation_end_sentence = ['!', '.', '?']
    #sentencesTimeds = []
    #print(len(wts))

    # parallel code
    sentencesTimeds = Parallel(n_jobs=-1)(delayed(process_one_file)(i, wt) for i, wt in enumerate(wts))

    '''for i, wt in enumerate(wts):
        print(i)
        sentencesTimed = pd.DataFrame(columns=['speaker', 'sentence_courante'])    

        st = sentenceTimed(wt)

        for word in range(len(wt)):
            if word == 0:
                st.modif_per_word(word)
            elif wt.iloc[word].word[0].isupper() and wt.iloc[word - 1].word in punctuation_end_sentence:
                st.modif_per_sentence(sentencesTimed, word)
            else:
                st.modif_per_word(word)

        st.add_sentence_informations_to_dataframe(sentencesTimed)
        sentencesTimeds.append(sentencesTimed)'''

    data = pd.concat(sentencesTimeds)

    X = [s.lower().split() for s in data.sentence_courante.values]
    Y = [s.lower() for s in data.speaker.values]

    #EMBEDDINGS PART
    #embed = nn.Embedding(num_embeddings, embedding_dim)
    # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
    #embed.weight.data.copy_(torch.from_numpy(pretrained_weight))

    #we = vocab.GloVe(name='6B', dim=100)
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

    to_del = []
    for s in X:
        for w in s:
            if w not in we.stoi:
                to_del.append(w)
    X = [[w for w in s if w not in to_del] for s in X]

    words_set = set()
    for s in X:
        words_set = words_set.union(set(s))

    threshold_train_dev = int(len(X)*0.8)
    threshold_dev_test = threshold_train_dev + int(len(X)*0.1)
    X_train = X[:threshold_train_dev]
    Y_train = Y[:threshold_train_dev]
    X_dev = X[threshold_train_dev:threshold_dev_test]
    Y_dev = Y[threshold_train_dev:threshold_dev_test]
    X_test = X[threshold_dev_test:]
    Y_test = Y[threshold_dev_test:]
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test, words_set, we