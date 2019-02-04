import data
import model

X_train, Y_train, X_dev, Y_dev, X_test, Y_test, words_set, we = data.load_data()
taille_embedding = len(we.vectors[we.stoi[X_train[0][0]]])
idx_set_words, embed, model_trained = model.launch_train(X_train, Y_train, words_set, we, taille_embedding, taille_context=3, bidirectional=False, num_layers=3, nb_epoch=5, targset_size=1)
model.get_prediction(X_train, Y_train, idx_set_words, embed, model_trained, taille_embedding, taille_context=3)
model.get_prediction(X_dev, Y_dev, idx_set_words, embed, model_trained, taille_embedding, taille_context=3)
model.get_prediction(X_test, Y_test, idx_set_words, embed, model_trained, taille_embedding, taille_context=3)