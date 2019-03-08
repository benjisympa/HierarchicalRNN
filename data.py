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

def load_data(path_transcripts='/vol/work2/galmant/transcripts/', type_sentence_embedding='lstm'):
    punctuations_end_sentence = ['.', '?', '!']

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
                if sentence and row[1]:
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
                X = [[w for w in s if w not in to_del] for s in X]
                for words_per_sentence in X:
                    words_set = words_set.union(set(words_per_sentence))
            else:
                X = X_
                Y = Y#_
            if len(X)>0 and len(Y)>0:
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
    return Y_new, max(nb_positives, nb_negatives)/len(newY)

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
    return seq_tensor

def split_by_context(X, Y, taille_context, batch_size, device='cpu'): #(8..,1,4096)
    list_tensors_X = []
    list_tensors_Y = []
    for i in range(X.shape[0] - (2*taille_context + 1)): #NOT 1 NOW 0 FOR THE NUMBER OF SENTENCES
        list_tensors_X.append(torch.index_select(X, 0, torch.tensor(list(range(i,i+2*(taille_context+1))), device=device))) #(8,1,4096)
        list_tensors_Y.append(torch.index_select(Y, 0, torch.tensor([taille_context], device=device))) #(1)
    tensor_split_X = torch.stack(list_tensors_X).transpose(0,1).view(2*(taille_context+1),-1,X.shape[-1]) #(n-8,8,1,4096) -> (8,n,1,4096) -> (8,n,4096)
    tensor_split_Y = torch.stack(list_tensors_Y).view(1,-1)#.squeeze(0) #.transpose(0,1) #(n,1) -> (1,n)
    minis_batch_X = {}
    minis_batch_Y = {}
    nb_batches = int(tensor_split_X.shape[1]/batch_size)
    for i in range(nb_batches):
        minis_batch_X[i] = tensor_split_X[:,i*batch_size:(i+1)*batch_size,:]
        minis_batch_Y[i] = tensor_split_Y[:,i*batch_size:(i+1)*batch_size]
    minis_batch_X[nb_batches] = tensor_split_X[:,nb_batches*batch_size:,:] #n/32 tensors of (8,32,4096)
    minis_batch_Y[nb_batches] = tensor_split_Y[:,nb_batches*batch_size:] #n/32 tensors of (1,32)
    shuffle_ids = sample(list(range(nb_batches+1)), k=nb_batches+1) #Shuffled inside an episode
    return [minis_batch_X[i] for i in shuffle_ids], [minis_batch_Y[i] for i in shuffle_ids]

def pre_calculate_features(X_all, Y_all, output_path, type_sentence_embedding, idx_set_words, embed, taille_context, batch_size, device='cpu'):
    # Concatenate all the datas in pytorch lists of tensors and create the mini-batch (8,32,4096) or (109,8*32,300)
    season_episode = 0
    poucentages_majority_class = []
    #inputs_embeddings = {}
    #outputs_refs = {}
    shuffle_ids_episodes = sample(list(range(len(X_all))), k=len(X_all)) #Shuffled between each episode
    for id_ in shuffle_ids_episodes:#zip(X_all,Y_all):
        X_ = X_all[id_]
        Y_ = Y_all[id_]
        print('file',season_episode+1,'on',len(X_all))
        if type_sentence_embedding == 'lstm':
            vectorized_seqs = [[idx_set_words[w] for w in s]for s in X_]
            words_embeddings = create_X(X_, vectorized_seqs, device)
            words_embeddings = embed(words_embeddings)
        else:
            infersent.build_vocab(X_, tokenize=True)
            words_embeddings = infersent.encode(X_, tokenize=True) #In fact it's sentences embeddings, just to have the same name !!! (B,D)
            words_embeddings = torch.from_numpy(words_embeddings).unsqueeze(1)# (B,D) -> (L,B,D)
            #words_embeddings = words_embeddings.to(device)
        #words_embeddings : (8..,1,4096) or (109,8..,300); 8.. -> number of sentences, 8 -> context size, 109 -> max number of words per sentence, 300 or 4096 -> embeddings size
        Y, poucentage_majority_class = create_Y(Y_, device) #(8..)
        #TODO CURRENTLY ONLY FOR INFERSENT
        inputs_embeddings_, outputs_refs_ = split_by_context(words_embeddings, Y, taille_context, batch_size, device=device) #(n/32,8,32,4096) and (n/32,1,32)
        #inputs_embeddings[season_episode] = inputs_embeddings_
        #outputs_refs[season_episode] = outputs_refs_
        
        if inputs_embeddings_.shape[0] > 0: #TODO il y a des tensors vide, par exemple le 142ème en partant de 0
            torch.save(inputs_embeddings_, output_path+'inputs_embeddings_'+str(season_episode)+'.pickle')
            torch.save(outputs_refs_, output_path+'outputs_refs_'+str(season_episode)+'.pickle')
        '''with open(output_path+'inputs_embeddings_'+str(season_episode)+'.pickle', 'wb') as handle:
            pickle.dump(inputs_embeddings_, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(output_path+'outputs_refs_'+str(season_episode)+'.pickle', 'wb') as handle:
            pickle.dump(outputs_refs_, handle, protocol=pickle.HIGHEST_PROTOCOL)'''
            
        #sentences_embeddings = sentence_embeddings_by_sum(words_embeddings, embed, vectorized_seqs)
        poucentages_majority_class.append(poucentage_majority_class)
        season_episode += 1
    
    '''shuffle_ids_episodes = sample(list(range(len(inputs_embeddings))), k=len(inputs_embeddings)) #Shuffled between each episode
    inputs_embeddings_shuffled = []
    outputs_refs_shuffled = []
    for i in shuffle_ids_episodes:
        inputs_embeddings_shuffled += inputs_embeddings[i]
        outputs_refs_shuffled += outputs_refs[i]'''
    
    #ids_episodes_shuffled = shuffle(range(len(inputs_embeddings)))
    print('mean poucentages majority class', sum(poucentages_majority_class)/len(poucentages_majority_class))

    '''with open(output_path+'inputs_embeddings.pickle', 'wb') as handle:
        pickle.dump(inputs_embeddings_shuffled, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_path+'outputs_refs.pickle', 'wb') as handle:
        pickle.dump(outputs_refs_shuffled, handle, protocol=pickle.HIGHEST_PROTOCOL)'''

def get_features(X_all, Y_all, output_path, words_set, we, taille_embedding, type_sentence_embedding, taille_context, batch_size, device='cpu'):
    idx_set_words = None
    embed = None
    if type_sentence_embedding == 'lstm':
        idx_set_words = dict(zip(list(words_set), range(1,len(words_set)+1)))
        idx_set_words['<PAD>'] = 0 #for padding we need to intialize one row of vector weights
        padding_idx = idx_set_words['<PAD>']
        embed = torch.nn.Embedding(num_embeddings=len(words_set)+1, embedding_dim=taille_embedding, padding_idx=padding_idx)
        we_idx = [0] #for padding we need to intialize one row of vector weights
        we_idx += [we.stoi[w] for w in list(words_set)]
        embed.weight.data.copy_(we.vectors[we_idx])
        embed.weight.requires_grad = False
        embed = embed.to(device)
        dim = 1
    else:
        dim = 0
    
    pre_calculate_features(X_all, Y_all, output_path, type_sentence_embedding, idx_set_words, embed, taille_context, batch_size, device=device)