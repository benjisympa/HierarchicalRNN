import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torchtext.vocab as vocab
import numpy as np
import matplotlib.pyplot as plt

# Set the random seed manually for reproducibility.
torch.manual_seed(1234)

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class HierarchicalBiLSTM_on_sentence_embedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, targset_size, num_layers = 3, bidirectional = False, device='cpu'):
        super(HierarchicalBiLSTM_on_sentence_embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 1
        if self.bidirectional:
            self.num_directions = 2
        self.num_layers = num_layers
        self.device = device

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm_previous = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False, bidirectional=bidirectional)
        self.lstm_previous = self.lstm_previous.to(device)
        self.lstm_future = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False, bidirectional=bidirectional)
        self.lstm_future = self.lstm_future.to(device)
        
        # Hidden state to hidden state
        self.linear_layers = []
        for _ in range(self.num_layers):
            hidden_layer = nn.Linear(2*self.num_directions*hidden_dim, 2*self.num_directions*hidden_dim, bias=True)
            self.linear_layers.append(hidden_layer.to(device))
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2*self.num_directions*hidden_dim, targset_size, bias=True)#2 because we concatenate the both output of the lstm
        self.hidden2tag = self.hidden2tag.to(device)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layers, 1, self.hidden_dim, device=self.device, requires_grad=False), torch.zeros(self.num_layers, 1, self.hidden_dim, device=self.device, requires_grad=False))

    def forward(self, input_previous_sentences, input_future_sentences):
        seq_tensor_output_previous, _ = self.lstm_previous(input_previous_sentences, self.hidden)
        seq_tensor_output_future, _ = self.lstm_future(input_future_sentences, self.hidden)
        
        seq_len = input_previous_sentences.shape[0]
        batch = input_previous_sentences.shape[1]

        seq_tensor_output_sum = torch.cat((seq_tensor_output_previous.view(seq_len, batch, self.num_directions, self.hidden_dim)[-1,:,:], seq_tensor_output_future.view(seq_len, batch, self.num_directions, self.hidden_dim)[-1,:,:]), -1)
        #TODO Attention mechanism
        seq_tensor_output_sum = seq_tensor_output_sum.view(batch,2*self.num_directions*self.hidden_dim) #2 is because we concatenate previous and future embeddings
        #lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        for layer in self.linear_layers:
            seq_tensor_output_sum = layer(seq_tensor_output_sum)
            seq_tensor_output_sum = torch.tanh(seq_tensor_output_sum)
        tag_space = self.hidden2tag(seq_tensor_output_sum)
        tag_space = tag_space[0]
        prediction = torch.sigmoid(tag_space)
        #print(tag_space, tag_space.clamp(min=0), torch.tanh(tag_space), prediction)
        return prediction

def create_Y(Y, device):
    newY = []
    for i in range(1, len(Y)):
        if Y[i] == Y[i-1]:
            newY.append([0])
        else:
            newY.append([1])
    Y_new = Variable(torch.FloatTensor(newY))
    Y_new = Y_new.to(device)
    return Y_new

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

def sentence_embeddings_by_sum(words_embeddings, embed, vectorized_seqs, taille_embedding):
    # embed your sequences
    seq_tensor = embed(words_embeddings)

    # sum over L, all words per sentence
    seq_tensor_sumed = torch.sum(seq_tensor, dim=0) #len(vectorized_seqs), taille_embedding
    #print('after sum', seq_tensor_sumed.shape)
    seq_tensor_sumed = seq_tensor_sumed.view(len(vectorized_seqs), 1, taille_embedding)
    #print(seq_tensor_sumed.shape)
    #print('sum', seq_tensor_sumed)
    return seq_tensor_sumed

def launch_train(X, Y, words_set, we, taille_embedding, taille_context=3, bidirectional=False, num_layers=3, nb_epoch=5, targset_size=1, device='cpu', is_trained=False):
    #https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e

    ### Premier modèle, somme des embedding par phrases

    idx_set_words = dict(zip(list(words_set), range(1,len(words_set)+1)))
    idx_set_words['<PAD>'] = 0 #for padding we need to intialize one row of vector weights
    padding_idx = idx_set_words['<PAD>']
    embed = nn.Embedding(num_embeddings=len(words_set)+1, embedding_dim=taille_embedding, padding_idx=padding_idx)
    we_idx = [0] #for padding we need to intialize one row of vector weights
    we_idx += [we.stoi[w] for w in list(words_set)]
    embed.weight.data.copy_(we.vectors[we_idx])
    embed.weight.requires_grad = False
    embed = embed.to(device)

    vectorized_seqs = [[idx_set_words[w] for w in s]for s in X]
    words_embeddings = create_X(X, vectorized_seqs, device)
    sentences_embeddings = sentence_embeddings_by_sum(words_embeddings, embed, vectorized_seqs, taille_embedding)
    Y = create_Y(Y, device)

    input_size = taille_embedding
    hidden_size = taille_embedding
    nb_sentences = len(vectorized_seqs)

    model = HierarchicalBiLSTM_on_sentence_embedding(taille_embedding, taille_embedding, targset_size, num_layers, bidirectional, device=device)
    model = model.to(device)
    loss_function = nn.BCELoss()#NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0005)
    
    if is_trained:
        return idx_set_words, embed, model, None
    else:
        print('début train')
        losses = []
        for epoch in range(nb_epoch):
            #get mini-batch
            #Data loader
            print('epoch', epoch)
            losses_ = []
            for i in range(nb_sentences - (2*taille_context + 1)):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                #print(i, nb_sentences, taille_context)
                model.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                model.hidden = model.init_hidden()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                indices_previous = torch.tensor(list(range(i,i+taille_context+1)), device=device)
                indices_future = torch.tensor(list(range(i+2*taille_context+1,i+taille_context,-1)), device=device)
                input_previous_features = torch.index_select(sentences_embeddings, 0, indices_previous)
                input_future_features = torch.index_select(sentences_embeddings, 0, indices_future)

                # Step 3. Run our forward pass.
                prediction = model(input_previous_features, input_future_features)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(prediction, Y[i+taille_context]) #targets)
                losses_.append(loss)
                loss.backward()
                optimizer.step()

                #break
            print(loss)
            losses.append(losses_)
            #break
        print('fin train')
        torch.save(model.state_dict(), '/people/maurice/HierarchicalRNN/last_model.pth.tar')
        return idx_set_words, embed, model, losses

def get_prediction(X, Y, idx_set_words, embed, model, taille_embedding, taille_context=3, device='cpu', is_eval=False):
    if is_eval:
        model_ = model.eval()
    else:
        model_ = model
    vectorized_seqs = [[idx_set_words[w] for w in s]for s in X]
    words_embeddings = create_X(X, vectorized_seqs, device)
    sentences_embeddings = sentence_embeddings_by_sum(words_embeddings, embed, vectorized_seqs, taille_embedding)
    Y = create_Y(Y, device)
    nb_sentences = len(vectorized_seqs)
    #print('nb_sentences', nb_sentences)
    
    # See what the scores are after training
    with torch.no_grad():
        error_global = 0
        iter_sentences = range(nb_sentences - (2*taille_context + 1))
        Y_positives = []
        Y_negatives = []
        for i in iter_sentences:
            indices_previous = torch.tensor(list(range(i,i+taille_context+1)), device=device)
            indices_future = torch.tensor(list(range(i+2*taille_context+1,i+taille_context,-1)), device=device)
            input_previous_features = torch.index_select(sentences_embeddings, 0, indices_previous)
            input_future_features = torch.index_select(sentences_embeddings, 0, indices_future)
            prediction = model_(input_previous_features, input_future_features).item()
            ref = Y[i+taille_context].item()
            if i < 30:
                print(prediction, ref, abs(ref - prediction))
            if ref == 1:
                Y_positives.append(prediction)
            elif ref == 0:
                Y_negatives.append(prediction)
            if abs(ref - prediction) >= 0.5:
                error_global += 1
        error_global /= len(iter_sentences)
        print('error_global', error_global)
        Y_positives = np.asarray(Y_positives)
        Y_negatives = np.asarray(Y_negatives)
        np.save('Y_positives.npy', Y_positives)
        np.save('Y_negatives.npy', Y_negatives)
        print(np.mean(Y_positives), np.mean(Y_negatives), np.mean(Y_positives) - np.mean(Y_negatives))