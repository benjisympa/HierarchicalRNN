import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torchtext.vocab as vocab
import numpy as np
import nltk
import math
import itertools
from random import shuffle
from random import sample
import pickle
import glob

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
        self.softmax = nn.Softmax(dim=1)
        
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

    def attention_vector(self, seq_tensor_output_p, seq_tensor_output_f, previous=True): #(4,32,hidden_size)
        similarity = torch.cat((seq_tensor_output_p[-1,:,:].repeat(seq_tensor_output_f.shape[0],1,1), seq_tensor_output_f[:,:,:]), 2)#(4,32,2*hidden_size)
        #print(similarity.shape) #torch.Size([4, 32, 600])
        if previous:
            similarity_ = self.tanh(self.attn_previous(similarity))#(4,32,10)
            #print(similarity_.shape) #torch.Size([4, 32, 10])
            alpha_ = self.alpha_previous(similarity_).transpose(0,1)#(32,4,1)
            #print(alpha_.shape) #torch.Size([32, 4, 1])
            alpha_ = self.softmax(alpha_)#(32,4,1)
            #print(alpha_.shape) #torch.Size([32, 4, 1])
            m = torch.matmul(seq_tensor_output_f.transpose(0,1).transpose(1,2), alpha_) #(32,hidden_size,4) * (32,4,1) -> (32,hidden_size,1)
            #print(m.shape) #torch.Size([300, 32, 1])
            #print(m.transpose(0,1).transpose(0,2).shape) #torch.Size([1, 32, 300])
        else:
            similarity_ = self.tanh(self.attn_future(similarity))#(4,32,10)
            alpha_ = self.alpha_future(similarity_).transpose(0,1)#(32,4,1)
            alpha_ = self.softmax(alpha_)
            m = torch.matmul(seq_tensor_output_p.transpose(0,1).transpose(1,2), alpha_) #(32,hidden_size,4) * (32,4,1) -> (32,hidden_size,1)
            #m = torch.matmul(seq_tensor_output_p.transpose(0,2), alpha_).transpose(0,1)
        return m.transpose(0,1).transpose(0,2) #torch.Size([1, 32, 300])

    def forward(self, sentences_emb):#(L,B,D) -> (109,8,300) avec lstm ou (B,L,D) -> (8,32,4096) avec pre-trained sentence embedding (infersent)
        if self.type_sentence_embedding == 'lstm':
            sentences_emb, self.hidden_sentences = self.lstm_sentence(sentences_emb, self.hidden_sentences)
            input_previous_sentences = sentences_emb[-1,:,:].unsqueeze(1)[:int(sentences_emb.shape[1]/2),:,:]#(B,L,D) -> (4,1,300) -> (L,B,D)
            input_future_sentences = sentences_emb[-1,:,:].unsqueeze(1)[int(sentences_emb.shape[1]/2):,:,:]#(B,L,D) -> (4,1,300) -> (L,B,D)
        else:
            input_previous_sentences = sentences_emb[:int(sentences_emb.shape[0]/2),:,:]#(B,L,D) -> (4,32,4096) -> (L,B,D)
            input_future_sentences = sentences_emb[int(sentences_emb.shape[0]/2):,:,:]#(B,L,D) -> (4,32,4096) -> (L,B,D)

        seq_tensor_output_previous, _ = self.lstm_previous(input_previous_sentences, self.hidden) #(4,32,hidden_size)
        seq_tensor_output_future, _ = self.lstm_future(input_future_sentences, self.hidden) #(4,32,hidden_size)

        seq_len = input_previous_sentences.shape[0]
        batch = input_previous_sentences.shape[1]

        #seq_tensor_output_sum = torch.cat((seq_tensor_output_previous.view(seq_len, batch, self.num_directions, self.hidden_dim)[-1,:,:], seq_tensor_output_future.view(seq_len, batch, self.num_directions, self.hidden_dim)[-1,:,:]), -1)
        #TODO Attention mechanism
        #seq_tensor_output_sum = seq_tensor_output_sum.view(batch,2*self.num_directions*self.hidden_dim) #2 is because we concatenate previous and future embeddings
        
        #TODO GERER LE BI-LSTM self.num_directions
        m_p = self.attention_vector(seq_tensor_output_previous, seq_tensor_output_future) #(4,32,hidden_size)
        m_f = self.attention_vector(seq_tensor_output_future, seq_tensor_output_previous, previous=False) #(4,32,hidden_size)
        
        #print(torch.unsqueeze(seq_tensor_output_previous[-1,:,:],0).shape, torch.unsqueeze(seq_tensor_output_future[-1,:,:],0).shape, m_p.shape, m_f.shape)#,seq_tensor_output_previous.shape,seq_tensor_output_future.shape)
        #torch.Size([1, 32, 300]) torch.Size([1, 32, 300]) torch.Size([1, 32, 300]) torch.Size([1, 32, 300]) torch.Size([4, 32, 300]) torch.Size([4, 32, 300])
        
        seq_tensor_output_sum = torch.cat((torch.unsqueeze(seq_tensor_output_previous[-1,:,:],0), torch.unsqueeze(seq_tensor_output_future[-1,:,:],0), m_p, m_f), -1) #(1,32,hidden_size) #TODO CONCATENER AUSSI LES 2 SENTENCES EMBEDDINGS CRITIQUES
        
        #lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        for layer in self.linear_layers:
            seq_tensor_output_sum = layer(seq_tensor_output_sum)
            seq_tensor_output_sum = torch.tanh(seq_tensor_output_sum)
        tag_space = self.hidden2tag(seq_tensor_output_sum) #(1,32,hidden_size) -> #(1,32,1)
        tag_space = tag_space[0] #(32,1)
        prediction = tag_space#torch.sigmoid(tag_space)
        #print(tag_space, tag_space.clamp(min=0), torch.tanh(tag_space), prediction)
        return prediction

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

def launch_train(model, path_model, path_data_train, path_data_dev, nb_epoch=5, device='cpu', type_sentence_embedding='lstm'):
    #https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e
    check_dev_epoch = 5
    
    '''with open(path_data_train+'inputs_embeddings.pickle', 'rb') as handle:
        inputs_embeddings_train = pickle.load(handle)
    with open(path_data_train+'outputs_refs.pickle', 'rb') as handle:
        outputs_refs_train = pickle.load(handle)
    with open(path_data_dev+'inputs_embeddings.pickle', 'rb') as handle:
        inputs_embeddings_dev = pickle.load(handle)
    with open(path_data_dev+'outputs_refs.pickle', 'rb') as handle:
        outputs_refs_dev = pickle.load(handle)'''

    #model = torch.nn.DataParallel(model, dim=dim)#, device_ids=[0, 1, 2])
    #pos_weight = torch.FloatTensor(len(negatives)/len(positives))
    #pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=None)#pos_weight)#BCELoss()#NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    #scheduler = StepLR(optimizer, step_size=math.ceil(nb_epoch/5), gamma=0.2)
    
    # Launch training
    print('début train')
    losses = []
    losses_dev = []
    #ids_iter=list(range(len(inputs_embeddings_train)))
    nb_files = len(list(glob.glob(path_data_train+'inputs_embeddings_*.pickle')))
    for epoch in range(nb_epoch):
        print('epoch',epoch+1,'on',nb_epoch)
        #shuffle(ids_iter)
        losses_ = []
        #scheduler.step(epoch)
        #for id_, it_ in enumerate(iter_):
        for it_ in range(nb_files):
            print(it_+1,'on',nb_files,'epoch',epoch+1,'on',nb_epoch)
            
            sentences_embs = torch.load(path_data_train+'inputs_embeddings_'+str(it_)+'.pickle')
            refs = torch.load(path_data_train+'outputs_refs_'+str(it_)+'.pickle')
            '''with open(path_data_train+'inputs_embeddings_'+str(it_)+'.pickle', 'rb') as handle:
                sentences_emb = pickle.load(handle)
            with open(path_data_train+'outputs_refs_'+str(it_)+'.pickle', 'rb') as handle:
                ref = pickle.load(handle)'''
            
            for sentences_emb, ref in zip(sentences_embs, refs): #Each file contains all the tensor of window-size for one episode
                #sentences_emb = inputs_embeddings_train[it_] #(8,32,4096)
                #ref = outputs_refs_train[it_] #(1,32)
                if sentences_emb.shape[0] == 0: #TODO il y a des tensors vide, par exemple le 142ème en partant de 0
                    continue
                sentences_emb = sentences_emb.to(device)
                ref = ref.to(device)
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                #print(i, nb_sentences)
                #model.zero_grad()
                # zero the parameter gradients
                optimizer.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                if type_sentence_embedding == 'lstm':
                    model.hidden_sentences = model.init_hidden(batch_size=sentences_emb.shape[1])
                model.hidden = model.init_hidden(batch_size=sentences_emb.shape[1])

                # Step 3. Run our forward pass.
                prediction = model(sentences_emb) #(32,1)    #(1,32,4096) or (109,8*32,300) ?

                prediction = torch.squeeze(prediction, 1)
                ref = torch.squeeze(ref, 0)
                #print(prediction.shape, ref.shape)
                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = criterion(prediction, ref) #targets)
                losses_.append(loss.item())
                loss.backward()
                optimizer.step()

            #break
        mean_loss_per_epoch = np.mean(np.asarray(losses_))
        print('Sum/len losses', sum(losses_)/len(losses_))
        print('Mean loss per epoch', mean_loss_per_epoch)
        losses.append(mean_loss_per_epoch)
        #model.get_prediction(X_, Y_, idx_set_words, embed, model, taille_context=taille_context, device=device)
        #break
        torch.save(model.state_dict(), path_model+'models/model_'+str(epoch)+'.pth.tar')
        #break
        if epoch%check_dev_epoch == 0:
            #ids_iter=list(range(len(inputs_embeddings_train)))
            losses_dev_ = []
            best_loss_dev = None
            for it_ in range(len(list(glob.glob(path_data_dev+'inputs_embeddings_*.pickle')))):
            
                sentences_emb = torch.load(path_data_dev+'inputs_embeddings_'+str(it_)+'.pickle')
                ref = torch.load(path_data_dev+'outputs_refs_'+str(it_)+'.pickle')
                '''with open(path_data_dev+'inputs_embeddings_'+str(it_)+'.pickle', 'rb') as handle:
                    sentences_emb = pickle.load(handle)
                with open(path_data_dev+'outputs_refs_'+str(it_)+'.pickle', 'rb') as handle:
                    ref = pickle.load(handle)'''
                
                for sentences_emb, ref in zip(sentences_embs, refs): #Each file contains all the tensor of window-size for one episode
                    #print(it_,'on',len(ids_iter),' dev')
                    #sentences_emb = inputs_embeddings_dev[it_] #(8,32,4096)
                    #ref = outputs_refs_dev[it_] #(1,32)
                    sentences_emb = sentences_emb.to(device)
                    ref = ref.to(device)

                    if type_sentence_embedding == 'lstm':
                        model.hidden_sentences = model.init_hidden(batch_size=sentences_emb.shape[1])
                    model.hidden = model.init_hidden(batch_size=sentences_emb.shape[1])

                    prediction = model(sentences_emb) #(32,1)    #(1,32,4096) or (109,8*32,300) ?
                    prediction = torch.squeeze(prediction, 1)
                    ref = torch.squeeze(ref, 0)

                    loss = criterion(prediction, ref) #targets)
                    losses_dev_.append(loss.item())
                    if not best_loss_dev or loss < best_loss_dev:
                        torch.save(model.state_dict(), path_model+'model_best_'+str(epoch)+'.pth.tar')
                        best_loss_dev = loss
            mean_loss_dev = np.mean(np.asarray(losses_dev_))
            print('Mean loss on dev',mean_loss_dev)
            losses_dev.append(mean_loss_dev)
            
    #torch.save(model.state_dict(), path_model+'model_best_'+str(epoch)+'.pth.tar')
    print('fin train')
    return losses

def get_predictions(model, subset, path_data, path_work, device='cpu', type_sentence_embedding='lstm'):  
    model = model.eval()
    
    #ids_iter=list(range(len(inputs_embeddings_train)))
    #ids_iter=list(range(len(inputs_embeddings_train)))
    #shuffle(ids_iter)
    # See what the scores are after training
    with torch.no_grad():
        error_global = 0
        Y_positives_all = []
        Y_negatives_all = []
        nb_files = len(list(glob.glob(path_data+'/'+subset+'/'+'inputs_embeddings_*.pickle')))
        for it_ in range(nb_files):
            print(it_+1,'on',nb_files,' prediction')
            
            sentences_embs = torch.load(path_data+'/'+subset+'/'+'inputs_embeddings_'+str(it_)+'.pickle')
            refs = torch.load(path_data+'/'+subset+'/'+'outputs_refs_'+str(it_)+'.pickle')
            '''with open(path_data+'/'+subset+'/'+'inputs_embeddings_'+str(it_)+'.pickle', 'rb') as handle:
                sentences_emb = pickle.load(handle)
            with open(path_data+'/'+subset+'/'+'outputs_refs_'+str(it_)+'.pickle', 'rb') as handle:
                ref = pickle.load(handle)'''
            
            for sentences_emb, ref in zip(sentences_embs, refs): #Each file contains all the tensor of window-size for one episode
                #sentences_emb = inputs_embeddings[it_] #(8,32,4096)
                #ref = outputs_refs[it_] #(1,32)
                #print(next(model.parameters()).is_cuda)
                sentences_emb = sentences_emb.to(device)
                ref = ref.to(device)

                if type_sentence_embedding == 'lstm':
                    model.hidden_sentences = model.init_hidden(batch_size=sentences_emb.shape[1])
                model.hidden = model.init_hidden(batch_size=sentences_emb.shape[1])
                prediction = model(sentences_emb)#.item()
                prediction = torch.squeeze(prediction, 1)
                ref = torch.squeeze(ref, 0)
                #print(prediction, ref, abs(ref - prediction))
                for i, v in enumerate(ref):
                    if np.asarray(v) == 1:
                        Y_positives_all.append(np.asarray(prediction[i]))
                    elif np.asarray(v) == 0:
                        Y_negatives_all.append(np.asarray(prediction[i]))

    Y_positives_all = np.asarray(Y_positives_all)
    Y_negatives_all = np.asarray(Y_negatives_all)
    np.save(path_work+'Y_positives_+'+subset+'.npy', Y_positives_all)
    np.save(path_work+'Y_negatives_+'+subset+'.npy', Y_negatives_all)
    print(np.mean(Y_positives_all), np.mean(Y_negatives_all), np.mean(Y_positives_all) - np.mean(Y_negatives_all))