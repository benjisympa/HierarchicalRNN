import data
import model
import pickle
import numpy as np
import torch

device = 'cuda:0'
is_trained = False
type_sentence_embedding='infersent'

X_train, Y_train, X_dev, Y_dev, X_test, Y_test, words_set, we = data.load_data(type_sentence_embedding=type_sentence_embedding)
hidden_size = 300
if type_sentence_embedding == 'lstm':
    taille_embedding = len(we.vectors[we.stoi[X_train[0][0][0]]])
else:
    taille_embedding = 4096

idx_set_words, embed, model_trained, losses = model.launch_train(X_train, Y_train, words_set, we, taille_embedding, hidden_size=hidden_size, taille_context=3, bidirectional=False, num_layers=1, nb_epoch=300, targset_size=1, device=device, is_trained=is_trained, type_sentence_embedding=type_sentence_embedding)
if is_trained:
    model_trained.load_state_dict(torch.load('/people/maurice/HierarchicalRNN/last_model.pth.tar'))
else:
    np.save('losses.npy', np.asarray(losses))

model.get_predictions(X_train, Y_train, idx_set_words, embed, model_trained, taille_context=3, device=device, is_eval=True, type_sentence_embedding=type_sentence_embedding)
#model.get_predictions(X_dev, Y_dev, idx_set_words, embed, model_trained, taille_context=3, device=device, is_eval=True, type_sentence_embedding=type_sentence_embedding)
#model.get_predictions(X_test, Y_test, idx_set_words, embed, model_trained, taille_context=3, device=device, is_eval=True, type_sentence_embedding=type_sentence_embedding)