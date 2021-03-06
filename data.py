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
import torch
from torch.autograd import Variable
from random import shuffle
from random import sample
from models import InferSent
from sklearn.model_selection import train_test_split
import sys
from threading import Thread

import torchtext.vocab as vocab
nlp = spacy.load('en')

# Set the random seed manually for reproducibility.
torch.manual_seed(1234)

V = 2
MODEL_PATH = '/vol/work3/maurice/encoder/infersent%s.pickle' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
W2V_PATH = '/vol/work3/maurice/dataset/fastText/crawl-300d-2M-subword.vec'
infersent.set_w2v_path(W2V_PATH)

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

def load_data(config, path_transcripts='/vol/work2/galmant/transcripts/'):
    type_sentence_embedding = config['type_sentence_embedding']
    dev_set_list = config['dev_set_list']
    test_set_list = config['test_set_list']
    
    punctuations_end_sentence = ['.', '?', '!']
    punctuations = string.punctuation #['!','(',')',',','-','.','/',':',';','<','=','>','?','[','\\',']','^','_','{','|','}','~'] #!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

    we = None
    if type_sentence_embedding == 'lstm':
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

    #X_all = []
    #Y_all = []
    X_train = []
    Y_train = []
    X_dev = []
    Y_dev = []
    X_test = []
    Y_test = []
    words_set = set()
    for file in sorted(glob.glob(path_transcripts+'*')):
        #TEST
        #for file in [sorted(glob.glob(path_transcripts+'*'))[0]]:
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            X_ = []
            Y_ = []
            for row in reader:
                sentence = row[2]
                old_word = row[2]
                for word in row[3:]:
                    if any(punctuation in old_word for punctuation in punctuations_end_sentence) and word and word[0].isupper():
                        sentence = sentence.strip()
                        n = 0
                        for i,s in enumerate(sentence):
                            if s in punctuations:
                                sentence_ = list(sentence)
                                sentence_.insert(i + n + 1,' ')
                                sentence_.insert(i + n,' ')
                                sentence = ''.join(sentence_)
                                n += 2
                        #print(sentence)
                        X_.append(sentence)
                        Y_.append(row[1])
                        sentence = word
                    else:
                        sentence += ' '+word
                    old_word = word
                if sentence and row[1]:
                    sentence = sentence.strip()
                    n = 0
                    for i,s in enumerate(sentence):
                        if s in punctuations:
                            sentence_ = list(sentence)
                            sentence_.insert(i + n + 1,' ')
                            sentence_.insert(i + n,' ')
                            sentence = ''.join(sentence_)
                            n += 2
                    #print(sentence)
                    X_.append(sentence)
                    Y_.append(row[1])
            Y = [s.lower() for s in Y_]
            if type_sentence_embedding == 'lstm':
                X = [s.lower().split() for s in X_]
                #Y = [s.lower() for s in Y_]
                to_del = []
                for s in X:
                    for w in s:
                        if w not in we.stoi:
                            to_del.append(w)
                X = [[w.strip() for w in s if w not in to_del] for s in X]
                for words_per_sentence in X:
                    words_set = words_set.union(set(words_per_sentence))
            else:
                X = X_
                Y = Y#_
            if len(X)>0 and len(Y)>0:
                names_episode = file.split('/')[-1]
                names_season = '.'.join(names_episode.split('.')[:-1])
                names_serie = '.'.join(names_episode.split('.')[0])
                if names_episode in dev_set_list or names_season in dev_set_list or names_serie in dev_set_list:
                    X_dev.append(X)
                    Y_dev.append(Y)
                elif names_episode in test_set_list or names_season in test_set_list or names_serie in test_set_list:
                    X_test.append(X)
                    Y_test.append(Y)
                else:
                    X_train.append(X)
                    Y_train.append(Y)
            assert len(X) == len(Y)

    '''threshold_train_dev = int(len(X_all)*0.8)
    threshold_dev_test = threshold_train_dev + int(len(X_all)*0.1)
    X_train = X_all[:threshold_train_dev]
    Y_train = Y_all[:threshold_train_dev]
    X_dev = X_all[threshold_train_dev:threshold_dev_test]
    Y_dev = Y_all[threshold_train_dev:threshold_dev_test]
    X_test = X_all[threshold_dev_test:]
    Y_test = Y_all[threshold_dev_test:]'''
    #TEST
    #X_train = X_test
    #Y_train = Y_test
    #X_dev = X_test
    #Y_dev = Y_test
    #print('X_train',X_train[-1])
    #time.sleep(60)
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test, words_set, we

def read_file(file, punctuations_end_sentence, words_set, dev_set_list, test_set_list, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, type_sentence_embedding, we):
    punctuations = string.punctuation
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        X_ = []
        Y_ = []
        for i_, row in enumerate(reader):
            if len(row) < 2:
                print(file, i_, row)
                continue
            sentence = row[1]
            old_word = row[1]
            for word in row[2:]:
                if any(punctuation in old_word for punctuation in punctuations_end_sentence) and word and word[0].isupper():
                    sentence = sentence.strip()
                    n = 0
                    for i,s in enumerate(sentence):
                        if s in punctuations:
                            sentence_ = list(sentence)
                            sentence_.insert(i + n + 1,' ')
                            sentence_.insert(i + n,' ')
                            sentence = ''.join(sentence_)
                            n += 2
                    #print(sentence)
                    X_.append(sentence)
                    Y_.append(row[0])
                    sentence = word
                else:
                    sentence += ' '+word
                old_word = word
            if sentence and row[0]:
                sentence = sentence.strip()
                n = 0
                for i,s in enumerate(sentence):
                    if s in punctuations:
                        sentence_ = list(sentence)
                        sentence_.insert(i + n + 1,' ')
                        sentence_.insert(i + n,' ')
                        sentence = ''.join(sentence_)
                        n += 2
                #print(sentence)
                X_.append(sentence)
                Y_.append(row[0])
        Y = [s.lower() for s in Y_]
        if type_sentence_embedding == 'lstm':
            X = [s.lower().split() for s in X_]
            #Y = [s.lower() for s in Y_]
            to_del = []
            for s in X:
                for w in s:
                    if w not in we.stoi:
                        to_del.append(w)
            X = [[w.strip() for w in s if w not in to_del] for s in X]
            for words_per_sentence in X:
                words_set = words_set.union(set(words_per_sentence))
        else:
            X = X_
            Y = Y#_
        if len(X)>0 and len(Y)>0:
            names_episode = file.split('/')[-1] #'TheBigBangTheory.Season01.Episode01'
            names_season = '.'.join(names_episode.split('.')[:-1]) #'TheBigBangTheory.Season01'
            names_serie = '.'.join(names_episode.split('.')[0]) #'TheBigBangTheory'
            names_season_etoile = '*.'+'.'.join(names_episode.split('.')[1:]) #'*.Season01.Episode01'
            names_serie_etoile = '*.'+'.'.join(names_episode.split('.')[1]) #'*.Season01'
            if names_episode in dev_set_list or names_season in dev_set_list or names_serie in dev_set_list or names_season_etoile in dev_set_list or names_serie_etoile in dev_set_list:
                X_dev.append(X)
                Y_dev.append(Y)
            elif names_episode in test_set_list or names_season in test_set_list or names_serie in test_set_list or names_season_etoile in test_set_list or names_serie_etoile in test_set_list:
                X_test.append(X)
                Y_test.append(Y)
            else:
                X_train.append(X)
                Y_train.append(Y)
        assert len(X) == len(Y)


def load_data_new(config, path_transcripts='/vol/work3/maurice/Transcripts/'):
    type_sentence_embedding = config['type_sentence_embedding']
    dev_set_list = config['dev_set_list']
    test_set_list = config['test_set_list']
    
    punctuations_end_sentence = ['.', '?', '!']
    punctuations = string.punctuation #['!','(',')',',','-','.','/',':',';','<','=','>','?','[','\\',']','^','_','{','|','}','~'] #!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

    we = None
    if type_sentence_embedding == 'lstm':
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

    #X_all = []
    #Y_all = []
    X_train = []
    Y_train = []
    X_dev = []
    Y_dev = []
    X_test = []
    Y_test = []
    words_set = set()
    threads = []
    for file in sorted(glob.glob(path_transcripts+'*/*')):
        #TEST
        #for file in [sorted(glob.glob(path_transcripts+'*'))[0]]:
        process = Thread(target=read_file, args=[file, punctuations_end_sentence, words_set, dev_set_list, test_set_list, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, type_sentence_embedding, we])
        process.start()
        threads.append(process)
    
    for process in threads:
        process.join()
    
    print(words_set)
        
    '''threshold_train_dev = int(len(X_all)*0.8)
    threshold_dev_test = threshold_train_dev + int(len(X_all)*0.1)
    X_train = X_all[:threshold_train_dev]
    Y_train = Y_all[:threshold_train_dev]
    X_dev = X_all[threshold_train_dev:threshold_dev_test]
    Y_dev = Y_all[threshold_train_dev:threshold_dev_test]
    X_test = X_all[threshold_dev_test:]
    Y_test = Y_all[threshold_dev_test:]'''
    #TEST
    #X_train = X_test
    #Y_train = Y_test
    #X_dev = X_test
    #Y_dev = Y_test
    #print('X_train',X_train[-1])
    #time.sleep(60)
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test, words_set, we

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def create_Y(Y, device):
    newY = []
    nb_positives = 0
    nb_negatives = 0
    for i in range(1, len(Y)):
        if Y[i] == Y[i-1]:
            newY.append([0])
            nb_negatives += 1
        else:
            newY.append([1])
            nb_positives += 1
    Y_new = Variable(torch.FloatTensor(newY))
    Y_new = Y_new.to(device)
    label_majority_class = 'Meme locuteur'
    if max(nb_positives, nb_negatives) == nb_positives:
        label_majority_class = 'Locuteur different'
    return Y_new, max(nb_positives, nb_negatives)/len(newY), label_majority_class

def create_X(X, vectorized_seqs, device):
    #dtype = torch.cuda.float if torch.cuda.is_available() else torch.float
    #torch.zeros(2, 2, dtype=dtype)

    # get the length of each seq in your batch
    seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
    #print('length', seq_lengths)
    # dump padding everywhere, and place seqs on the left.
    # NOTE: you only need a tensor as big as your longest sequence
    seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

    #print(seq_tensor[0:2])
    #print('nb sentences', len(vectorized_seqs))

    # utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
    # Otherwise, give (L,B,D) tensors
    seq_tensor = seq_tensor.transpose(0,1) # (B,L,D) -> (L,B,D)
    seq_tensor = seq_tensor.to(device)
    return seq_tensor, seq_lengths

import time
def split_by_context(config, X, X_lengths, Y, shuffled=False, device='cpu'): #(8..,1,4096)
    taille_context = config['taille_context']
    batch_size = config['batch_size']
    list_tensors_X = []
    list_tensors_Y = []
    #X 357 torch.Size([34, 357, 300]) Il y a 357 phrases de 34 mots max par phrase
    #print(X.size(), Y.size()) #torch.Size([34, 357, 300]) torch.Size([356, 1])
    minis_batch_X = {}
    minis_batch_X_length = {}
    minis_batch_Y = {}
    if config['type_sentence_embedding'] == 'infersent':
        for i in range(X.shape[0] - (2*taille_context + 1) + 1): #NOT 1 NOW 0 FOR THE NUMBER OF SENTENCES # NEW + 1 dans le range
            list_tensors_X.append(torch.index_select(X, 0, torch.tensor(list(range(i,i+2*(taille_context+1))), device=device))) #(8,1,4096)
            list_tensors_Y.append(torch.index_select(Y, 0, torch.tensor([i+taille_context], device=device))) #(1)
        #list_tensors_X[0] torch.Size([8, 34, 300])
        #torch.stack(list_tensors_X).transpose(0,1) torch.Size([8, 350, 34, 300])
        tensor_split_X = torch.stack(list_tensors_X).transpose(0,1).view(2*(taille_context+1),-1,X.shape[-1]) #(n-8,8,1,4096) -> (8,n,1,4096) -> (8,n,4096)
        tensor_split_Y = torch.stack(list_tensors_Y).view(1,-1)#.squeeze(0) #.transpose(0,1) #(n,1) -> (1,n)
        nb_batches = int(tensor_split_X.shape[1]/batch_size)
        for i in range(nb_batches):
            minis_batch_X[i] = tensor_split_X[:,i*batch_size:(i+1)*batch_size,:]
            minis_batch_Y[i] = tensor_split_Y[:,i*batch_size:(i+1)*batch_size]
        minis_batch_X[nb_batches] = tensor_split_X[:,nb_batches*batch_size:,:] #n/32 tensors of (8,32,4096)
        minis_batch_Y[nb_batches] = tensor_split_Y[:,nb_batches*batch_size:] #n/32 tensors of (1,32)
    else:
        #print(Y[0]) #tensor([1.])
        Y = Y.transpose(0,1) #torch.Size([356])
        #print(Y.size()) #torch.Size([1, 340])
        nb_batches = int(X.shape[1]/batch_size)
        nb_batches_ = nb_batches
        for i in range(nb_batches):
            minis_batch_X[i] = X[:,i*batch_size:(i+1)*batch_size,:]
            minis_batch_X_length[i] = X_lengths[i*batch_size:(i+1)*batch_size]
            minis_batch_Y[i] = Y[:,i*batch_size:(i+1)*batch_size - 1]
        if X.shape[1] - nb_batches*batch_size > 0:
            minis_batch_X[nb_batches] = X[:,X.shape[1]-batch_size:,:] #n/32 tensors of (8,32,4096)
            minis_batch_X_length[nb_batches] = X_lengths[X.shape[1]-batch_size:]
            minis_batch_Y[nb_batches] = Y[:,Y.shape[1]-batch_size + 1:] #n/32 tensors of (1,32)
            nb_batches_ += 1
        '''if X.shape[1]-(nb_batches*batch_size) >= 2*(taille_context+1):
            minis_batch_X[nb_batches] = X[:,nb_batches*batch_size:,:] #n/32 tensors of (8,32,4096)
            minis_batch_Y[nb_batches] = Y[:,nb_batches*batch_size:] #n/32 tensors of (1,32)
        else:
            minis_batch_X[nb_batches] = X[:,nb_batches*batch_size-(2*(taille_context+1)-(X.shape[1]-(nb_batches*batch_size))):,:] #n/32 tensors of (8,32,4096)
            minis_batch_Y[nb_batches] = Y[:,nb_batches*batch_size-(2*(taille_context+1)-(X.shape[1]-(nb_batches*batch_size))):] #n/32 tensors of (1,32)'''
        #print(X[:,nb_batches*batch_size:,:].size(), Y[:,nb_batches*batch_size:].size()) #torch.Size([49, 6, 300]) torch.Size([1, 5])
    #print('minis_batch_X[0]',minis_batch_X[0])
    #print('X',X[:,-1,:])
    #time.sleep(60)
    shuffle_ids = list(range(nb_batches_))
    if shuffled:
        shuffle_ids = sample(list(range(nb_batches+1)), k=nb_batches+1) #Shuffled inside an episode
    return [minis_batch_X[i] for i in shuffle_ids], [minis_batch_X_length[i] for i in shuffle_ids], [minis_batch_Y[i] for i in shuffle_ids] #(L,B,D) -> (n/32,109,32,300)=(n/32,109,8,300) 109: nb de mots max par phrase

def pre_calculate_features(config, X_all, Y_all, output_path, idx_set_words, embed, shuffled=False, device='cpu'):
    # Concatenate all the datas in pytorch lists of tensors and create the mini-batch (8,32,4096) or (109,8*32,300)
    season_episode = 0
    poucentages_majority_class = []
    #inputs_embeddings = {}
    #outputs_refs = {}
    shuffle_ids_episodes = list(range(len(X_all)))
    if shuffled:
        shuffle_ids_episodes = sample(list(range(len(X_all))), k=len(X_all)) #Shuffled between each episode
    for id_ in shuffle_ids_episodes:#zip(X_all,Y_all):
        X_ = X_all[id_]
        Y_ = Y_all[id_]
        print('file',season_episode+1,'on',len(X_all))
        #X_lengths = None
        if config['type_sentence_embedding'] == 'lstm':
            #print('X_',X_)
            #vectorized_seqs = [[idx_set_words[w] for w in s]for s in X_]
            vectorized_seqs = []
            for s in X_:
                swe = [idx_set_words[w] for w in s]
                if len(swe) > 0:
                    vectorized_seqs.append(swe)
            #X_lengths = [len(sentence) for sentence in X_]
            #print('vectorized_seqs', vectorized_seqs)
            words_embeddings, X_lengths = create_X(X_, vectorized_seqs, device)
            #print('words_embeddings id', words_embeddings)
            words_embeddings = embed(words_embeddings)
            #print('words_embeddings', words_embeddings)
        else:
            infersent.build_vocab(X_, tokenize=True)
            words_embeddings = infersent.encode(X_, tokenize=True) #In fact it's sentences embeddings, just to have the same name !!! (B,D)
            words_embeddings = torch.from_numpy(words_embeddings).unsqueeze(1)# (B,D) -> (L,B,D)
            #words_embeddings = words_embeddings.to(device)
        #words_embeddings : (8..,1,4096) or (109,8..,300); 8.. -> number of sentences, 8 -> context size, 109 -> max number of words per sentence, 300 or 4096 -> embeddings size
        Y, poucentage_majority_class, label_majority_class = create_Y(Y_, device) #(8..)
        #TODO CURRENTLY ONLY FOR INFERSENT
        inputs_embeddings_, X_lengths_, outputs_refs_ = split_by_context(config, words_embeddings, X_lengths, Y, shuffled=shuffled, device=device) #(n/32,8,32,4096) and (n/32,1,32)
        #inputs_embeddings[season_episode] = inputs_embeddings_
        #outputs_refs[season_episode] = outputs_refs_
        
        #if inputs_embeddings_.shape[0] > 0: #TODO il y a des tensors vide, par exemple le 142ème en partant de 0
        #Il faudrait itérer sur chacun des tenseurs de inputs_embeddings_ et les supprimer de la liste si ils sont vides
        torch.save(inputs_embeddings_, output_path+'inputs_embeddings_'+str(season_episode)+'.pickle')
        torch.save(X_lengths_, output_path+'X_lengths_'+str(season_episode)+'.pickle')
        torch.save(outputs_refs_, output_path+'outputs_refs_'+str(season_episode)+'.pickle')
        '''with open(output_path+'inputs_embeddings_'+str(season_episode)+'.pickle', 'wb') as handle:
            pickle.dump(inputs_embeddings_, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(output_path+'outputs_refs_'+str(season_episode)+'.pickle', 'wb') as handle:
            pickle.dump(outputs_refs_, handle, protocol=pickle.HIGHEST_PROTOCOL)'''
            
        #sentences_embeddings = sentence_embeddings_by_sum(words_embeddings, embed, vectorized_seqs)
        poucentages_majority_class.append(poucentage_majority_class)
        season_episode += 1
        
        print('Majority class per episode', label_majority_class, poucentage_majority_class)
    
    '''shuffle_ids_episodes = sample(list(range(len(inputs_embeddings))), k=len(inputs_embeddings)) #Shuffled between each episode
    inputs_embeddings_shuffled = []
    outputs_refs_shuffled = []
    for i in shuffle_ids_episodes:
        inputs_embeddings_shuffled += inputs_embeddings[i]
        outputs_refs_shuffled += outputs_refs[i]'''
    
    #ids_episodes_shuffled = shuffle(range(len(inputs_embeddings)))
    print('Global mean poucentages majority class', sum(poucentages_majority_class)/len(poucentages_majority_class))

    '''with open(output_path+'inputs_embeddings.pickle', 'wb') as handle:
        pickle.dump(inputs_embeddings_shuffled, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_path+'outputs_refs.pickle', 'wb') as handle:
        pickle.dump(outputs_refs_shuffled, handle, protocol=pickle.HIGHEST_PROTOCOL)'''

def get_embds(config, words_set, we, device='cpu'):
    idx_set_words = dict(zip(list(words_set), range(1,len(words_set)+1)))
    idx_set_words['<PAD>'] = 0 #for padding we need to intialize one row of vector weights
    padding_idx = idx_set_words['<PAD>']
    embed = torch.nn.Embedding(num_embeddings=len(words_set)+1, embedding_dim=config['taille_embedding'], padding_idx=padding_idx)
    we_idx = [0] #for padding we need to intialize one row of vector weights
    we_idx += [we.stoi[w] for w in list(words_set)]
    embed.weight.data.copy_(we.vectors[we_idx])
    embed.weight.requires_grad = False
    embed = embed.to(device)
    return idx_set_words, embed

#OLD not used anymore
def get_features(config, X_all, Y_all, output_path, words_set, we, shuffled=False, device='cpu'):
    idx_set_words = None
    embed = None
    if config['type_sentence_embedding'] == 'lstm':
        idx_set_words = dict(zip(list(words_set), range(1,len(words_set)+1)))
        idx_set_words['<PAD>'] = 0 #for padding we need to intialize one row of vector weights
        padding_idx = idx_set_words['<PAD>']
        embed = torch.nn.Embedding(num_embeddings=len(words_set)+1, embedding_dim=config['taille_embedding'], padding_idx=padding_idx)
        we_idx = [0] #for padding we need to intialize one row of vector weights
        we_idx += [we.stoi[w] for w in list(words_set)]
        embed.weight.data.copy_(we.vectors[we_idx])
        embed.weight.requires_grad = False
        embed = embed.to(device)
        dim = 1
    else:
        dim = 0
        
    return idx_set_words, embed
    #pre_calculate_features(config, X_all, Y_all, output_path, idx_set_words, embed, shuffled=shuffled, device=device)
