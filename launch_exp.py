import data
import model as Model
import pickle
import numpy as np
import torch
import pathlib
import glob

use_pre_trained_features = True
launch_train = True

device = 'cpu'
if launch_train:
    device = 'cuda:0'
pre_load_data = False
type_sentence_embedding='infersent'
subset_data = 'big_bang_theory_:'
path_save = '/vol/work3/maurice/HierarchicalRNN/'
hidden_size = 300
batch_size = 32
taille_context = 3
targset_size = 1
num_layers = 1
nb_epoch = 500
bidirectional = False

if type_sentence_embedding == 'infersent':
    taille_embedding = 4096

'''
Data will be store like this :

path_save/type_sentence_embedding/subset_data/models/pytorch_model_epoch_0.pth.tar
path_save/type_sentence_embedding/subset_data/pytorch_best_model_epoch_0.pth.tar
path_save/type_sentence_embedding/subset_data/pre_trained_features/train/inputs_embeddings_0.pickle
path_save/type_sentence_embedding/subset_data/pre_trained_features/train/outputs_refs_0.pickle
path_save/type_sentence_embedding/subset_data/pre_trained_features/dev/inputs_embeddings_0.pickle
path_save/type_sentence_embedding/subset_data/pre_trained_features/dev/outputs_refs_0.pickle
path_save/type_sentence_embedding/subset_data/pre_trained_features/test/inputs_embeddings_0.pickle
path_save/type_sentence_embedding/subset_data/pre_trained_features/test/outputs_refs_0.pickle
path_save/type_sentence_embedding/subset_data/work/outputs_refs_0.pickle

Where type_sentence_embedding is one of : lstm, infersent, ...
and subset_data if one of : big_bang_theory_:, big_bang_theory_season_1:3, big_bang_theory_:_game_of_thrones_::
'''

#Create paths if not exists
print('Create paths')
sub_path = path_save+type_sentence_embedding+'/'+subset_data
path_model = sub_path+'/'
path_data = sub_path+'/pre_trained_features'
path_work = sub_path+'/work/'

pathlib.Path(sub_path+'/models/').mkdir(parents=True, exist_ok=True)
pathlib.Path(path_data+'/train/').mkdir(parents=True, exist_ok=True)
pathlib.Path(path_data+'/dev/').mkdir(parents=True, exist_ok=True)
pathlib.Path(path_data+'/test/').mkdir(parents=True, exist_ok=True)
pathlib.Path(path_work).mkdir(parents=True, exist_ok=True)

if not use_pre_trained_features:
    #Load all the Dataset/Corpus
    print('Load corpus dataset')
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test, words_set, we = data.load_data(type_sentence_embedding=type_sentence_embedding)
    if type_sentence_embedding == 'lstm':
        taille_embedding = len(we.vectors[we.stoi[X_train[0][0][0]]])

    #Calculate features on CPU
    print('Calculate features on CPU')
    data.get_features(X_train, Y_train, path_data+'/train/', words_set, we, taille_embedding, type_sentence_embedding, taille_context, batch_size, device='cpu')
    data.get_features(X_dev, Y_dev, path_data+'/dev/', words_set, we, taille_embedding, type_sentence_embedding, taille_context, batch_size, device='cpu')
    data.get_features(X_test, Y_test, path_data+'/test/', words_set, we, taille_embedding, type_sentence_embedding, taille_context, batch_size, device='cpu')

#Define the model
print('Define the model')
model = Model.HierarchicalBiLSTM_on_sentence_embedding(taille_embedding, hidden_size, targset_size, num_layers, bidirectional=bidirectional, device=device, type_sentence_embedding=type_sentence_embedding)
model = model.to(device)

if launch_train:
    #Launch train on GPU
    print('Launch train on GPU')
    losses = Model.launch_train(model, path_model, path_data+'/train/', path_data+'/dev/', nb_epoch=nb_epoch, device=device, type_sentence_embedding=type_sentence_embedding)
    np.save(path_work+'losses.npy', np.asarray(losses))

#To use model
subset = 'test'
print('Predict on test')
best_model_path = list(glob.glob(path_model+'model_best_*.pth.tar'))[0]
model.load_state_dict(torch.load(best_model_path))
Model.get_predictions(model, subset, path_data, path_work, device='cpu', type_sentence_embedding=type_sentence_embedding)

#Model.get_predictions(X_train, Y_train, embed, model_trained, batch_size=32, taille_context=3, device=device, is_eval=True, type_sentence_embedding=type_sentence_embedding)
#model.get_predictions(X_dev, Y_dev, idx_set_words, embed, model_trained, taille_context=3, device=device, is_eval=True, type_sentence_embedding=type_sentence_embedding)
#model.get_predictions(X_test, Y_test, idx_set_words, embed, model_trained, taille_context=3, device=device, is_eval=True, type_sentence_embedding=type_sentence_embedding)