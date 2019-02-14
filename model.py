import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torchtext.vocab as vocab
import numpy as np
import matplotlib.pyplot as plt
import nltk
from models import InferSent

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

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class HierarchicalBiLSTM_on_sentence_embedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=300, targset_size=1, num_layers=3, hidden_linear_size=10, bidirectional=False, device='cpu', type_sentence_embedding='lstm'):
        super(HierarchicalBiLSTM_on_sentence_embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 1
        if self.bidirectional:
            self.num_directions = 2
        self.num_layers = num_layers
        self.device = device
        self.type_sentence_embedding = type_sentence_embedding

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if type_sentence_embedding == 'lstm':
            self.lstm_sentence = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False, bidirectional=bidirectional)
            self.lstm_sentence = self.lstm_sentence.to(device)
            size_sentence_embedding = hidden_dim*num_directions
        else:
            size_sentence_embedding = embedding_dim
        self.lstm_previous = torch.nn.LSTM(input_size=size_sentence_embedding, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False, bidirectional=bidirectional)
        self.lstm_previous = self.lstm_previous.to(device)
        self.lstm_future = torch.nn.LSTM(input_size=size_sentence_embedding, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False, bidirectional=bidirectional)
        self.lstm_future = self.lstm_future.to(device)
        
        # Attention Layer
        self.attn_previous = nn.Linear(2*hidden_dim,hidden_linear_size)
        self.attn_previous = self.attn_previous.to(device)
        self.alpha_previous = nn.Linear(hidden_linear_size,1)
        self.alpha_previous = self.alpha_previous.to(device)
        self.attn_future = nn.Linear(2*hidden_dim,hidden_linear_size)
        self.attn_future = self.attn_future.to(device)
        self.alpha_future = nn.Linear(hidden_linear_size,1)
        self.alpha_future = self.alpha_future.to(device)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        
        # Hidden state to hidden state
        self.linear_layers = []
        '''for _ in range(self.num_layers):
            hidden_layer = nn.Linear(2*self.num_directions*hidden_dim, 2*self.num_directions*hidden_dim, bias=True)
            self.linear_layers.append(hidden_layer.to(device))'''
        self.linear_layers.append(nn.Linear(4*self.num_directions*hidden_dim, 2*self.num_directions*hidden_dim, bias=True).to(device))
        self.linear_layers.append(nn.Linear(2*self.num_directions*hidden_dim, self.num_directions*hidden_dim, bias=True).to(device))
        self.linear_layers.append(nn.Linear(self.num_directions*hidden_dim, self.num_directions*hidden_dim, bias=True).to(device))
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(self.num_directions*hidden_dim, targset_size, bias=True)#2 because we concatenate the both output of the lstm
        self.hidden2tag = self.hidden2tag.to(device)
        self.hidden_sentences = self.init_hidden()
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_dim, device=self.device), torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_dim, device=self.device))#, requires_grad=False))
    
    def attention_vector(self, seq_tensor_output_p, seq_tensor_output_f, previous=True):
        similarity = torch.cat((seq_tensor_output_p[-1,:,:].repeat(seq_tensor_output_f.shape[0],1,1), seq_tensor_output_f[:,:,:]), 2)
        if previous:
            similarity_ = self.tanh(self.attn_previous(similarity))
            alpha_ = self.alpha_previous(similarity_).squeeze()
        else:
            similarity_ = self.tanh(self.attn_future(similarity))
            alpha_ = self.alpha_future(similarity_).squeeze()
        alpha_ = self.softmax(alpha_)
        m = torch.matmul(seq_tensor_output_f.transpose(0,2), alpha_).transpose(0,1)
        return m

    def forward(self, sentences_emb):#(L,B,D) -> (109,8,300) avec lstm ou (B,L,D) -> (8,1,4096) avec pre-trained sentence embedding (infersent)
        if self.type_sentence_embedding == 'lstm':
            sentences_emb, self.hidden_sentences = self.lstm_sentence(sentences_emb, self.hidden_sentences)
            input_previous_sentences = sentences_emb[-1,:,:].unsqueeze(1)[:int(sentences_emb.shape[1]/2),:,:]#(B,L,D) -> (4,1,300) -> (L,B,D)
            input_future_sentences = sentences_emb[-1,:,:].unsqueeze(1)[int(sentences_emb.shape[1]/2):,:,:]#(B,L,D) -> (4,1,300) -> (L,B,D)
        else:
            input_previous_sentences = sentences_emb[:int(sentences_emb.shape[0]/2),:,:]#(B,L,D) -> (4,1,4096) -> (L,B,D)
            input_future_sentences = sentences_emb[int(sentences_emb.shape[0]/2):,:,:]#(B,L,D) -> (4,1,4096) -> (L,B,D)
        
        seq_tensor_output_previous, _ = self.lstm_previous(input_previous_sentences, self.hidden)
        seq_tensor_output_future, _ = self.lstm_future(input_future_sentences, self.hidden)
        
        seq_len = input_previous_sentences.shape[0]
        batch = input_previous_sentences.shape[1]

        #seq_tensor_output_sum = torch.cat((seq_tensor_output_previous.view(seq_len, batch, self.num_directions, self.hidden_dim)[-1,:,:], seq_tensor_output_future.view(seq_len, batch, self.num_directions, self.hidden_dim)[-1,:,:]), -1)
        #TODO Attention mechanism
        #seq_tensor_output_sum = seq_tensor_output_sum.view(batch,2*self.num_directions*self.hidden_dim) #2 is because we concatenate previous and future embeddings
        
        #TODO GERER LE BI-LSTM self.num_directions
        m_p = self.attention_vector(seq_tensor_output_previous, seq_tensor_output_future)
        m_f = self.attention_vector(seq_tensor_output_future, seq_tensor_output_previous, previous=False)
        
        seq_tensor_output_sum = torch.cat((seq_tensor_output_previous[-1,:,:], seq_tensor_output_future[-1,:,:], m_p, m_f), -1) #TODO CONCATENER AUSSI LES 2 SENTENCES EMBEDDINGS CRITIQUES
        
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

def sentence_embeddings_by_sum(words_embeddings, embed, vectorized_seqs):
    # embed your sequences
    seq_tensor = words_embeddings#embed(words_embeddings)

    # sum over L, all words per sentence
    seq_tensor_sumed = torch.sum(seq_tensor, dim=0) #len(vectorized_seqs), taille_embedding
    #print('after sum', seq_tensor_sumed.shape)
    #seq_tensor_sumed = seq_tensor_sumed.view(len(vectorized_seqs), 1, taille_embedding)
    #print(seq_tensor_sumed.shape)
    #print('sum', seq_tensor_sumed)
    return seq_tensor_sumed

def launch_train(X_all, Y_all, words_set, we, taille_embedding, hidden_size=300, taille_context=3, bidirectional=False, num_layers=3, nb_epoch=5, targset_size=1, device='cpu', is_trained=False, type_sentence_embedding='lstm'):
    #https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e

    ### Premier modèle, somme des embedding par phrases

    idx_set_words = None
    embed = None
    if type_sentence_embedding == 'lstm':
        idx_set_words = dict(zip(list(words_set), range(1,len(words_set)+1)))
        idx_set_words['<PAD>'] = 0 #for padding we need to intialize one row of vector weights
        padding_idx = idx_set_words['<PAD>']
        embed = nn.Embedding(num_embeddings=len(words_set)+1, embedding_dim=taille_embedding, padding_idx=padding_idx)
        we_idx = [0] #for padding we need to intialize one row of vector weights
        we_idx += [we.stoi[w] for w in list(words_set)]
        embed.weight.data.copy_(we.vectors[we_idx])
        embed.weight.requires_grad = False
        embed = embed.to(device)
    
    model = HierarchicalBiLSTM_on_sentence_embedding(taille_embedding, hidden_size, targset_size, num_layers, bidirectional=bidirectional, device=device, type_sentence_embedding=type_sentence_embedding)
    model = model.to(device)
    criterion = nn.BCELoss()#NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=int(nb_epoch/5), gamma=0.2)

    if is_trained:
        return idx_set_words, embed, model, None
    
    print('début train')
    season_episode = 0
    losses = []
    poucentages_majority_class = []
    for X_,Y_ in zip(X_all,Y_all):
        print('file',season_episode,'on',len(X_all))
        if type_sentence_embedding == 'lstm':
            vectorized_seqs = [[idx_set_words[w] for w in s]for s in X_]
            words_embeddings = create_X(X_, vectorized_seqs, device)
            words_embeddings = embed(words_embeddings)
        else:
            infersent.build_vocab(X_, tokenize=True)
            words_embeddings = infersent.encode(X_, tokenize=True) #In fact it's sentences embeddings, just to have the same name !!! (B,D)
            words_embeddings = torch.from_numpy(words_embeddings).unsqueeze(0)# (B,D) -> (L,B,D)
            words_embeddings = words_embeddings.to(device)
        #sentences_embeddings = sentence_embeddings_by_sum(words_embeddings, embed, vectorized_seqs)
        Y, poucentage_majority_class = create_Y(Y_, device)
        poucentages_majority_class.append(poucentage_majority_class)

        nb_sentences = words_embeddings.shape[1]#len(vectorized_seqs)

        for epoch in range(nb_epoch):
            #get mini-batch
            #Data loader
            print('epoch',epoch,'on',nb_epoch,'file',season_episode,'on',len(X_all))
            losses_ = []
            scheduler.step(epoch)
            for i in range(nb_sentences - (2*taille_context + 1)):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                #print(i, nb_sentences, taille_context)
                #model.zero_grad()
                # zero the parameter gradients
                optimizer.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                sentences_emb = torch.index_select(words_embeddings, 1, torch.tensor(list(range(i,i+2*taille_context+2)), device=device))#(L,B,D) -> (*,n,d)

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                if type_sentence_embedding == 'lstm':
                    model.hidden_sentences = model.init_hidden(batch_size=sentences_emb.shape[1])
                else:
                    sentences_emb = sentences_emb.transpose(0,1)#(B,L,D) -> (8,1,300)
                model.hidden = model.init_hidden()

                # Step 3. Run our forward pass.
                prediction = model(sentences_emb)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = criterion(prediction, Y[i+taille_context]) #targets)
                losses_.append(loss.item())
                loss.backward()
                optimizer.step()

                #break
            print(sum(losses_)/len(losses_))
            losses.append(losses_)
            #model.get_prediction(X_, Y_, idx_set_words, embed, model, taille_context=taille_context, device=device)
            #break
        season_episode += 1
        torch.save(model.state_dict(), '/people/maurice/HierarchicalRNN/last_model.pth.tar')
        break
    print('fin train')
    print('mean poucentages majority class', sum(poucentages_majority_class)/len(poucentages_majority_class))
    return idx_set_words, embed, model, losses

def get_one_prediction(X, Y, idx_set_words, embed, model, taille_context=3, device='cpu', is_eval=False, type_sentence_embedding='lstm'):
    if is_eval:
        model = model.eval()
    if type_sentence_embedding == 'lstm':
        vectorized_seqs = [[idx_set_words[w] for w in s]for s in X]
        words_embeddings = create_X(X, vectorized_seqs, device)
        words_embeddings = embed(words_embeddings)
    else:
        infersent.build_vocab(X, tokenize=True)
        words_embeddings = infersent.encode(X, tokenize=True) #In fact it's sentences embeddings, just to have the same name !!! (B,D)
        words_embeddings = torch.from_numpy(words_embeddings).unsqueeze(0)# (B,D) -> (L,B,D)
        words_embeddings = words_embeddings.to(device)
    #sentences_embeddings = sentence_embeddings_by_sum(words_embeddings, embed, vectorized_seqs)
    Y, _ = create_Y(Y, device)
    nb_sentences = words_embeddings.shape[1]#len(vectorized_seqs)
    #print('nb_sentences', nb_sentences)
    
    # See what the scores are after training
    with torch.no_grad():
        error_global = 0
        iter_sentences = range(nb_sentences - (2*taille_context + 1))
        Y_positives = []
        Y_negatives = []
        for i in iter_sentences:
            sentences_emb = torch.index_select(words_embeddings, 1, torch.tensor(list(range(i,i+2*taille_context+2)), device=device))
            if type_sentence_embedding == 'lstm':
                model.hidden_sentences = model.init_hidden(batch_size=sentences_emb.shape[1])
            else:
                sentences_emb = sentences_emb.transpose(0,1)#(B,L,D) -> (8,1,300)
            model.hidden = model.init_hidden()
            prediction = model(sentences_emb).item()
            ref = Y[i+taille_context].item()
            if i < 5:
                print(prediction, ref, abs(ref - prediction))
            if ref == 1:
                Y_positives.append(prediction)
            elif ref == 0:
                Y_negatives.append(prediction)
        return Y_positives, Y_negatives

def get_predictions(X_all, Y_all, idx_set_words, embed, model, taille_context=3, device='cpu', is_eval=False, type_sentence_embedding='lstm'):
    Y_positives_all = []
    Y_negatives_all = []
    file = 0
    for X,Y in zip(X_all,Y_all):
        print('file', file, 'on', len(X_all))
        Y_positives, Y_negatives = get_one_prediction(X, Y, idx_set_words, embed, model, taille_context=taille_context, device=device, is_eval=is_eval,type_sentence_embedding=type_sentence_embedding)
        Y_positives_all += Y_positives
        Y_negatives_all += Y_negatives
        file += 1
    Y_positives_all = np.asarray(Y_positives_all)
    Y_negatives_all = np.asarray(Y_negatives_all)
    np.save('Y_positives.npy', Y_positives_all)
    np.save('Y_negatives.npy', Y_negatives_all)
    print(np.mean(Y_positives_all), np.mean(Y_negatives_all), np.mean(Y_positives_all) - np.mean(Y_negatives_all))