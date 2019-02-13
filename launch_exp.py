import data
import model
import pickle
import numpy as np
import torch

device = 'cuda:0'
is_trained = False

X_train, Y_train, X_dev, Y_dev, X_test, Y_test, words_set, we = data.load_data()
taille_embedding = len(we.vectors[we.stoi[X_train[0][0]]])

idx_set_words, embed, model_trained, losses = model.launch_train(X_train, Y_train, words_set, we, taille_embedding, taille_context=3, bidirectional=False, num_layers=1, nb_epoch=100, targset_size=1, device=device, is_trained=is_trained)
if is_trained:
    model_trained.load_state_dict(torch.load('/people/maurice/HierarchicalRNN/last_model.pth.tar'))
else:
    np.save('losses.npy', np.asarray(losses))

model.get_prediction(X_train, Y_train, idx_set_words, embed, model_trained, taille_embedding, taille_context=3, device=device, is_eval=True)
#model.get_prediction(X_dev, Y_dev, idx_set_words, embed, model_trained, taille_embedding, taille_context=3, device=device, is_eval=True)
#model.get_prediction(X_test, Y_test, idx_set_words, embed, model_trained, taille_embedding, taille_context=3, device=device, is_eval=True)