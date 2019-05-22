import data
import model as Model
import pickle
import numpy as np
import torch
import pathlib
import glob

use_pre_trained_features = False#True
launch_train = True
restart_at_epoch = 0#996

#Reproductibility
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1234)

config = {}

config['dev_set_list']=['TheBigBangTheory.Season02']
config['test_set_list']=['TheBigBangTheory.Season01']
config['device'] = 'cpu'
gpu = 'cuda:1'#0
if launch_train:
    config['device'] = gpu
#pre_load_data = False
config['type_sentence_embedding']='lstm'#'infersent'
subset_data = 'big_bang_theory_:'
path_save = '/vol/work3/maurice/HierarchicalRNN/'
config['hidden_size'] = 300
config['batch_size'] = 32 #Should be a multiple of the taille_context for the LSTM model
config['taille_context'] = 3
config['targset_size'] = 1
config['hidden_linear_size'] = 10
config['num_layers'] = 1
config['dp_ratio'] = 0.5
config['nb_epoch'] = 10000
config['bidirectional'] = False
exp_name = 'context_'+str(config['taille_context'])+'_hidden-size_'+str(config['hidden_size'])+'_hidden-linear-size_'+str(config['hidden_linear_size'])+'_num-layers_'+str(config['num_layers'])+'_dp-ratio_'+str(config['dp_ratio'])+'_bidirectional_'+str(config['bidirectional'])

if config['type_sentence_embedding'] == 'infersent':
    config['taille_embedding'] = 4096

'''
Data will be store like this :

path_save/config['type_sentence_embedding']/subset_data/exp_name/models/pytorch_model_epoch_0.pth.tar
path_save/config['type_sentence_embedding']/subset_data/exp_name/pytorch_best_model_epoch_0.pth.tar
path_save/config['type_sentence_embedding']/subset_data/exp_name/pre_trained_features/train/inputs_embeddings_0.pickle
path_save/config['type_sentence_embedding']/subset_data/exp_name/pre_trained_features/train/outputs_refs_0.pickle
path_save/config['type_sentence_embedding']/subset_data/exp_name/pre_trained_features/dev/inputs_embeddings_0.pickle
path_save/config['type_sentence_embedding']/subset_data/exp_name/pre_trained_features/dev/outputs_refs_0.pickle
path_save/config['type_sentence_embedding']/subset_data/exp_name/pre_trained_features/test/inputs_embeddings_0.pickle
path_save/config['type_sentence_embedding']/subset_data/exp_name/pre_trained_features/test/outputs_refs_0.pickle
path_save/config['type_sentence_embedding']/subset_data/exp_name/work/outputs_refs_0.pickle

Where config['type_sentence_embedding'] is one of : lstm, infersent, ...
and subset_data if one of : big_bang_theory_:, big_bang_theory_season_1:3, big_bang_theory_:_game_of_thrones_::
'''

#Create paths if not exists
print('Create paths')
sub_path = path_save+config['type_sentence_embedding']+'/'+subset_data+'/'+exp_name
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
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test, words_set, we = data.load_data(config)
    if config['type_sentence_embedding'] == 'lstm':
        config['taille_embedding'] = len(we.vectors[we.stoi[X_train[0][0][0]]])

    #Calculate features on CPU
    print('Calculate features on CPU')
    config['device'] = 'cpu'
    data.get_features(config, X_train, Y_train, path_data+'/train/', words_set, we, shuffled=False)#True)
    data.get_features(config, X_dev, Y_dev, path_data+'/dev/', words_set, we, shuffled=False)
    data.get_features(config, X_test, Y_test, path_data+'/test/', words_set, we, shuffled=False)
else:
    config['taille_embedding'] = 300

#Define the model
print('Define the model')
config['device'] = gpu
model = Model.HierarchicalBiLSTM_on_sentence_embedding(config)
model = model.to(config['device'])
if restart_at_epoch > 0:
    model.load_state_dict(torch.load(path_model+'model_best_'+str(restart_at_epoch-1)+'.pth.tar'))

if launch_train:
    #Launch train on GPU
    print('Launch train on GPU')
    losses = Model.launch_train(model, path_model, path_data+'/train/', path_data+'/dev/', nb_epoch=config['nb_epoch'], device=config['device'], type_sentence_embedding=config['type_sentence_embedding'], restart_at_epoch=restart_at_epoch)
    np.save(path_work+'losses.npy', np.asarray(losses))

#Re-define the model on CPU
config['device'] = 'cpu'
model = Model.HierarchicalBiLSTM_on_sentence_embedding(config)
model = model.to(config['device'])

#To use model
subset = 'test'
print('Predict on test')
best_model_path = list(glob.glob(path_model+'model_best_*.pth.tar'))[-1]
model.load_state_dict(torch.load(best_model_path))
Model.get_predictions(config, model, subset, path_data, path_work, save=True)

#Model.get_predictions(X_train, Y_train, embed, model_trained, is_eval=True, config)
#model.get_predictions(X_dev, Y_dev, idx_set_words, embed, model_trained, is_eval=True, config)
#model.get_predictions(X_test, Y_test, idx_set_words, embed, model_trained, is_eval=True, config)
