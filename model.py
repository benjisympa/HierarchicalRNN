import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from tqdm import tqdm
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


torch.cuda.manual_seed_all(1234)

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def sort_sequences(inputs, lengths):
    """sort_sequences
    Sort sequences according to lengths descendingly.

    :param inputs (Tensor): input sequences, size [L, B, D]
    :param lengths (Tensor): length of each sequence, size [B]
    """
    lengths_sorted, sorted_idx = lengths.sort(descending=True)
    _, unsorted_idx = sorted_idx.sort()
    return inputs[:, sorted_idx, :], lengths_sorted, unsorted_idx

class HierarchicalBiLSTM_on_sentence_embedding(nn.Module):
    def __init__(self, config):
        super(HierarchicalBiLSTM_on_sentence_embedding, self).__init__()
        
        self.embedding_dim = config['taille_embedding']
        self.hidden_dim = config['hidden_size']
        self.bidirectional = config['bidirectional']
        self.dp_ratio = config['dp_ratio']
        self.taille_context = 2*(1+config['taille_context'])
        self.num_directions = 1
        if self.bidirectional:
            self.num_directions = 2
        self.num_layers = config['num_layers']
        self.device = config['device']
        self.type_sentence_embedding = config['type_sentence_embedding']
        self.target_size = config['target_size']
        self.hidden_linear_size = config['hidden_linear_size']

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.type_sentence_embedding == 'lstm':
            self.lstm_sentence = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=False, bidirectional=self.bidirectional, dropout=self.dp_ratio)
            self.lstm_sentence = self.lstm_sentence.to(self.device)
            size_sentence_embedding = self.hidden_dim*self.num_directions
        else:
            size_sentence_embedding = self.embedding_dim
        self.lstm_previous = torch.nn.LSTM(input_size=size_sentence_embedding, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=False, bidirectional=self.bidirectional, dropout=self.dp_ratio)
        self.lstm_previous = self.lstm_previous.to(self.device)
        self.lstm_future = torch.nn.LSTM(input_size=size_sentence_embedding, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=False, bidirectional=self.bidirectional, dropout=self.dp_ratio)
        self.lstm_future = self.lstm_future.to(self.device)
        
        # Attention Layer
        self.attn_previous = nn.Linear(2*self.hidden_dim,self.hidden_linear_size)
        self.attn_previous = self.attn_previous.to(self.device)
        self.alpha_previous = nn.Linear(self.hidden_linear_size,1)
        self.alpha_previous = self.alpha_previous.to(self.device)
        self.attn_future = nn.Linear(2*self.hidden_dim,self.hidden_linear_size)
        self.attn_future = self.attn_future.to(self.device)
        self.alpha_future = nn.Linear(self.hidden_linear_size,1)
        self.alpha_future = self.alpha_future.to(self.device)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
        self.dropout = nn.Dropout(p=self.dp_ratio)
        self.relu = nn.ReLU()
        
        # Hidden state to hidden state
        self.linear_layers = []
        '''for _ in range(self.num_layers):
            hidden_layer = nn.Linear(2*self.num_directions*hidden_dim, 2*self.num_directions*hidden_dim, bias=True)
            self.linear_layers.append(hidden_layer.to(device))'''
        self.linear_layers.append(nn.Linear(4*self.num_directions*self.hidden_dim, 2*self.num_directions*self.hidden_dim, bias=True).to(self.device))
        self.linear_layers.append(nn.Linear(2*self.num_directions*self.hidden_dim, self.num_directions*self.hidden_dim, bias=True).to(self.device))
        self.linear_layers.append(nn.Linear(self.num_directions*self.hidden_dim, self.num_directions*self.hidden_dim, bias=True).to(self.device))
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(self.num_directions*self.hidden_dim, self.target_size, bias=True)#2 because we concatenate the both output of the lstm
        self.hidden2tag = self.hidden2tag.to(self.device)
        self.hidden_sentences = self.init_hidden()
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_dim, device=self.device)), Variable(torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_dim, device=self.device)))#, requires_grad=False))

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
            #print(seq_tensor_output_p.transpose(0,1).transpose(1,2).size(), alpha_.size()) #torch.Size([1, 300, 28]) torch.Size([1, 4, 1]) ???
            m = torch.matmul(seq_tensor_output_p.transpose(0,1).transpose(1,2), alpha_) #(32,hidden_size,4) * (32,4,1) -> (32,hidden_size,1)
            #m = torch.matmul(seq_tensor_output_p.transpose(0,2), alpha_).transpose(0,1)
        return m.transpose(0,1).transpose(0,2) #torch.Size([1, 32, 300])

    def forward_sentence(self, sentences_emb, X_length):
    #sentences_emb -> (L,B,D) : nb de mots max par phrase, nombre de phrases, dimension des words embeddings
    #def forward(self, inputs, lengths=None, hidden=None):
        #print('ENTREE FONCTION FORWARD')
        if X_length is not None:
            sentences_emb_p, sorted_lengths, unsorted_idx = sort_sequences(sentences_emb, X_length)
            sentences_emb_p = torch.nn.utils.rnn.pack_padded_sequence(sentences_emb_p, sorted_lengths, batch_first=False)

        sentences_emb_, (ht, ct) = self.lstm_sentence(sentences_emb_p, self.hidden_sentences)
        
        if X_length is not None:
            sentences_emb_, _ = torch.nn.utils.rnn.pad_packed_sequence(sentences_emb_, batch_first=False)
            sentences_emb_ = sentences_emb_.index_select(1, unsorted_idx)
            ht = ht.index_select(1, unsorted_idx)
            ct = ct.index_select(1, unsorted_idx)
            self.hidden_sentences = (ht, ct)
        
        X_length_1 = X_length-torch.ones((len(X_length)), dtype=torch.long, device=self.device)
        #sélectionne la dernière sentence embedding avant d'être un vecteur de 0, c'est à dire que l'on sélectionne en fonction de la taille de chaque phrase sur tout le batch
        sentences_emb_n = Variable(torch.zeros((sentences_emb_.shape[1], sentences_emb_.shape[2]), dtype=torch.float32, device=self.device))#, requires_grad=True)
        for idx in range(len(X_length_1)):
            sentences_emb_n[idx, :] = sentences_emb_[X_length_1[idx], idx, :]
        
        '''
        print(X_length)
        print('*****')
        print(X_length-torch.ones((len(X_length)), dtype=torch.long, device=self.device))
        print('*****')
        print(sentences_emb)
        print('*****')
        print(sentences_emb_)
        print('*****')
        print(sentences_emb.shape, sentences_emb_.shape)
        print(sentences_emb_[0])
        print(sentences_emb_[0].shape)
        print(sentences_emb_[-1,:,:])
        print('*****')
        print(sentences_emb_[5]) #
        print('*****')
        print(sentences_emb_[6])
        print('*****')
        print(sentences_emb_[7])
        print('*****')
        X_length_1 = X_length-torch.ones((len(X_length)), dtype=torch.long, device=self.device)
        print(sentences_emb_.shape, X_length_1.shape)
        print('*****')
        #sélectionne la dernière sentence embedding avant d'être un vecteur de 0, c'est à dire que l'on sélectionne en fonction de la taille de chaque phrase sur tout le batch
        sentences_emb_n = Variable(torch.zeros((sentences_emb_.shape[1], sentences_emb_.shape[2]), dtype=torch.float64, device=self.device), requires_grad=True)
        for idx in range(len(X_length_1)):
            #print(sentences_emb_[X_length_1[idx], idx, :])
            sentences_emb_n[idx, :] = sentences_emb_[X_length_1[idx], idx, :]
            #print(sentences_emb_n[idx, :])
        print(sentences_emb_n)
        print(sentences_emb_n.shape)
        '''
        return sentences_emb_n #(32,300)
    
    #torch.Size([8, 25, 300]) NEW
    def forward(self, sentences_emb):#(L,B,D) -> (109,32/8..,300) avec lstm ou (B,L,D) -> (8,32,4096) avec pre-trained sentence embedding (infersent)        
        input_previous_sentences = sentences_emb[:int(sentences_emb.shape[0]/2),:,:]#(B,L,D) -> (4,32,4096) -> (L,B,D)
        input_future_sentences = sentences_emb[int(sentences_emb.shape[0]/2):,:,:]#(B,L,D) -> (4,32,4096) -> (L,B,D)
        #print('input_previous_sentences',input_previous_sentences)
        #print('input_future_sentences',input_future_sentences)

        #print(self.hidden[0].size(), input_previous_sentences.size()) #torch.Size([1, 4, 300]) torch.Size([4, 24, 300])
        seq_tensor_output_previous, _ = self.lstm_previous(input_previous_sentences, self.hidden) #(4,32,hidden_size)
        seq_tensor_output_future, _ = self.lstm_future(input_future_sentences, self.hidden) #(4,32,hidden_size)
        #print('seq_tensor_output_previous',seq_tensor_output_previous)
        #print('seq_tensor_output_future',seq_tensor_output_future)
        
        #print(input_previous_sentences.size(), input_future_sentences.size()) #torch.Size([4, 4, 300]) torch.Size([4, 4, 300])
        #print(seq_tensor_output_previous.size(), seq_tensor_output_future.size())#torch.Size([4, 4, 300]) torch.Size([4, 4, 300]) #torch.Size([4, 1, 300]) torch.Size([28, 1, 300])

        seq_len = input_previous_sentences.shape[0]
        batch = input_previous_sentences.shape[1]

        #seq_tensor_output_sum = torch.cat((seq_tensor_output_previous.view(seq_len, batch, self.num_directions, self.hidden_dim)[-1,:,:], seq_tensor_output_future.view(seq_len, batch, self.num_directions, self.hidden_dim)[-1,:,:]), -1)
        #TODO Attention mechanism
        #seq_tensor_output_sum = seq_tensor_output_sum.view(batch,2*self.num_directions*self.hidden_dim) #2 is because we concatenate previous and future embeddings
        
        #TODO GERER LE BI-LSTM self.num_directions
        #print(seq_tensor_output_previous.size(), seq_tensor_output_future.size())
        m_p = self.attention_vector(seq_tensor_output_previous, seq_tensor_output_future) #(4,32,hidden_size)
        m_f = self.attention_vector(seq_tensor_output_future, seq_tensor_output_previous, previous=False) #(4,32,hidden_size)
        #print('m_p',m_p)
        #print('m_f',m_f)
        
        #print(torch.unsqueeze(seq_tensor_output_previous[-1,:,:],0).shape, torch.unsqueeze(seq_tensor_output_future[-1,:,:],0).shape, m_p.shape, m_f.shape)#,seq_tensor_output_previous.shape,seq_tensor_output_future.shape)
        #torch.Size([1, 32, 300]) torch.Size([1, 32, 300]) torch.Size([1, 32, 300]) torch.Size([1, 32, 300]) torch.Size([4, 32, 300]) torch.Size([4, 32, 300])
        
        seq_tensor_output_sum = torch.cat((torch.unsqueeze(seq_tensor_output_previous[-1,:,:],0), torch.unsqueeze(seq_tensor_output_future[-1,:,:],0), m_p, m_f), -1) #(1,32,4*hidden_size) #TODO CONCATENER AUSSI LES 2 SENTENCES EMBEDDINGS CRITIQUES
        #print('seq_tensor_output_sum',seq_tensor_output_sum)
        
        #print(torch.unsqueeze(seq_tensor_output_previous[-1,:,:],0).size(), torch.unsqueeze(seq_tensor_output_future[-1,:,:],0).size(), m_p.size(), m_f.size())
        #torch.Size([1, 4, 300]) torch.Size([1, 4, 300]) torch.Size([1, 4, 300]) torch.Size([1, 4, 300])
        #lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        for layer in self.linear_layers:
            seq_tensor_output_sum = layer(seq_tensor_output_sum)
            #print('seq_tensor_output_sum linear',seq_tensor_output_sum)
            #seq_tensor_output_sum = self.dropout(self.relu(seq_tensor_output_sum)) #nn.functional.glu()
            seq_tensor_output_sum = torch.tanh(seq_tensor_output_sum)
            #print('seq_tensor_output_sum linear tanh',seq_tensor_output_sum)
        tag_space = self.hidden2tag(seq_tensor_output_sum) #(1,32,hidden_size) -> #(1,32,1)
        #print(seq_tensor_output_sum.size(), tag_space.size(), tag_space[0].size()) #torch.Size([1, 4, 300]) torch.Size([1, 4, 1]) torch.Size([4, 1])
        #print('tag_space',tag_space)
        tag_space = tag_space[0] #(32,1)
        #print('tag_space',tag_space)
        prediction = torch.sigmoid(tag_space)#tag_space
        #print('prediction',prediction)
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

def launch_train(config, model, path_model, path_data_train, path_data_dev, nb_epoch=5, device='cpu', type_sentence_embedding='lstm', restart_at_epoch=0):
    #https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e
    check_dev_epoch = 1

    writer = SummaryWriter(comment='1 couche')
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
    criterion = nn.BCELoss()#nn.BCEWithLogitsLoss(pos_weight=None)#pos_weight)#BCELoss()#NLLLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #scheduler = StepLR(optimizer, step_size=math.ceil(nb_epoch/5), gamma=0.2)
    #scheduler = ReduceLROnPlateau(optimizer, 'min')
    
    model = model.train()
    
    # Launch training
    print('début train')
    losses = []
    losses_dev = []
    #ids_iter=list(range(len(inputs_embeddings_train)))
    nb_files = len(list(glob.glob(path_data_train+'inputs_embeddings_*.pickle')))
    for epoch in range(restart_at_epoch, nb_epoch):
        print('epoch',epoch+1,'on',nb_epoch)
        #shuffle(ids_iter)
        losses_ = []
        #scheduler.step(epoch)
        #TEST
        #for id_, it_ in enumerate(iter_):
        Y_pred = []
        Y_ref = []
        for it_ in tqdm(range(nb_files)):
        #for it_ in range(nb_files):
            #print(it_+1,'on',nb_files,'epoch',epoch+1,'on',nb_epoch)
            
            sentences_embs = torch.load(path_data_train+'inputs_embeddings_'+str(it_)+'.pickle')
            X_lengths = torch.load(path_data_train+'X_lengths_'+str(it_)+'.pickle')
            refs = torch.load(path_data_train+'outputs_refs_'+str(it_)+'.pickle')
            '''with open(path_data_train+'inputs_embeddings_'+str(it_)+'.pickle', 'rb') as handle:
                sentences_emb = pickle.load(handle)
            with open(path_data_train+'outputs_refs_'+str(it_)+'.pickle', 'rb') as handle:
                ref = pickle.load(handle)'''
            
            for sentences_emb, X_length, ref in zip(sentences_embs, X_lengths, refs): #Each file contains all the tensor of window-size for one episode
                #sentences_emb = inputs_embeddings_train[it_] #(8,32,4096)
                #ref = outputs_refs_train[it_] #(1,32)
                if sentences_emb.shape[0] == 0: #TODO il y a des tensors vide, par exemple le 142ème en partant de 0
                    #print('tensor empty wtf')
                    continue
                sentences_emb = sentences_emb.to(device)
                X_length = X_length.to(device)
                ref = ref.to(device)
                #torch.Size([36, 32, 300]) torch.Size([1, 31])
                #print(sentences_emb.size(), ref.size()) #torch.Size([34, 32, 300]) torch.Size([1, 31])
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
                    model.hidden_sentences = model.init_hidden(batch_size=sentences_emb.shape[1]) #(L,B,D) -> (109,8..,300)
                    #model.hidden = model.init_hidden(batch_size=int(sentences_emb.shape[1]/model.taille_context))
                    # Step 3. Run our forward pass.
                    #print('sentences_emb word embeddings',sentences_emb)
                    sentences_emb_ = model.forward_sentence(sentences_emb, X_length) #(32,300)
                    #print('sentences_emb_',sentences_emb_.shape)
                    #print('sentences_emb_',sentences_emb_)
                    to_packed_X = []
                    to_packed_Y = []
                    ref = ref.squeeze(0)
                    #print(sentences_emb_.shape[0], model.taille_context, sentences_emb_.shape[0] - model.taille_context, ref.size())
                    for i in range(sentences_emb_.shape[0] - model.taille_context + 1):
                        to_packed_X.append(torch.index_select(sentences_emb_, 0, torch.tensor(list(range(i,i+model.taille_context)), device=device)))
                        to_packed_Y.append(torch.index_select(ref, 0, torch.tensor([i+(int(model.taille_context/2)-1)], device=device)))
                    sentences_emb = torch.stack(to_packed_X).transpose(0,1) #(n,8,300) -> (8,n,300)
                    sentences_emb = sentences_emb.to(device)
                    ref = torch.stack(to_packed_Y).transpose(0,1) #(n,1) -> (1,n)
                    #torch.Size([8, 25, 300]) torch.Size([1, 25])
                    #print(sentences_emb.size(), ref.size())
                    
                model.hidden = model.init_hidden(batch_size=sentences_emb.shape[1])
                    
                # Step 3. Run our forward pass.
                #print('sentences_emb',sentences_emb.shape)
                #print('sentences_emb',sentences_emb)
                prediction = model(sentences_emb) #(32,1)    #(1,32,4096) or (109,8*32..,300)
                
                #WTFFFF torch.Size([25, 1]) torch.Size([1, 25]) torch.Size([8, 25, 300])
                #print('WTFFFF', prediction.size(), ref.size(), sentences_emb.size()) #torch.Size([4, 1]) torch.Size([1, 32]) torch.Size([34, 32, 300])
                prediction = torch.squeeze(prediction, 1)
                ref = torch.squeeze(ref, 0)
                #tensor([0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659, 0.6659], device='cuda:1', grad_fn=<SqueezeBackward1>) tensor([1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1.], device='cuda:1')
                #print(prediction, ref)
                #print(prediction.shape, ref.shape)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                #print(prediction.size(), ref.size(), sentences_emb.size()) #torch.Size([4]) torch.Size([356, 1]) torch.Size([34, 32, 300])
                loss = criterion(prediction, ref) #targets)
                losses_.append(loss.item())
                loss.backward()
                optimizer.step()

                # To calculate the EER
                Y_pred.append(np.asarray(prediction.detach().to('cpu')))
                Y_ref.append(np.asarray(ref.to('cpu')))
            #break
        '''# Calculate the EER
        model_eval = model.eval()
        fpr, tpr, threshold = roc_curve(Y_ref, Y_pred, pos_label=1)
        fnr = 1 - tpr
        eer_threshold = threshold(np.nanargmin(np.absolute((fnr - fpr))))
        eer = fpr(np.nanargmin(np.absolute((fnr - fpr))))'''
        #print(np.mean(np.concatenate(Y_ref, axis=None)), np.mean(np.concatenate(Y_pred, axis=None)))
        Y_ref = np.concatenate(Y_ref, axis=None)
        Y_pred = np.concatenate(Y_pred, axis=None)
        writer.add_pr_curve('score_train', np.mean(Y_ref), np.mean(Y_pred), epoch)
        mean_loss_train = np.mean(np.asarray(losses_))
        
        mean_loss_per_epoch = mean_loss_train#np.mean(np.asarray(losses_))
        #print('Sum/len losses', sum(losses_)/len(losses_))
        print('Mean loss per epoch train', mean_loss_per_epoch)
        losses.append(mean_loss_per_epoch)
        #model.get_prediction(X_, Y_, idx_set_words, embed, model, taille_context=taille_context, device=device)
        #break
        #TEST gagne du temps en ne sauvegardant pas les modèles
        torch.save(model.state_dict(), path_model+'models/model_'+str(epoch)+'.pth.tar')
        #break
        if epoch%check_dev_epoch == 0: #We evaluate on dev set
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

            # Calculate the EER
            #model_eval = model.eval() #TODO
            #Y_ref = np.concatenate(Y_ref, axis=None)
            #Y_pred = np.concatenate(Y_pred, axis=None)
            fpr, tpr, threshold = roc_curve(Y_ref, Y_pred, pos_label=1)
            fnr = 1 - tpr
            eer_threshold_train = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
            eer_train = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            
            #ids_iter=list(range(len(inputs_embeddings_train)))
            losses_dev_ = []
            Y_pred = []
            Y_ref = []
            best_loss_dev = None
            id_best_loss_dev = 0
            aa = True
            for it_ in range(len(list(glob.glob(path_data_dev+'inputs_embeddings_*.pickle')))):
            
                sentences_emb = torch.load(path_data_dev+'inputs_embeddings_'+str(it_)+'.pickle')
                X_lengths = torch.load(path_data_dev+'X_lengths_'+str(it_)+'.pickle')
                ref = torch.load(path_data_dev+'outputs_refs_'+str(it_)+'.pickle')
                '''with open(path_data_dev+'inputs_embeddings_'+str(it_)+'.pickle', 'rb') as handle:
                    sentences_emb = pickle.load(handle)
                with open(path_data_dev+'outputs_refs_'+str(it_)+'.pickle', 'rb') as handle:
                    ref = pickle.load(handle)'''
                
                for sentences_emb, ref in zip(sentences_embs, refs): #Each file contains all the tensor of window-size for one episode
                    #print(it_,'on',len(ids_iter),' dev')
                    #sentences_emb = inputs_embeddings_dev[it_] #(8,32,4096)
                    #ref = outputs_refs_dev[it_] #(1,32)
                    if sentences_emb.shape[0] == 0: #TODO il y a des tensors vide, par exemple le 142ème en partant de 0
                        print('tensor empty wtf')
                        continue
                    sentences_emb = sentences_emb.to(device)
                    X_length = X_length.to(device)
                    ref = ref.to(device)

                    if type_sentence_embedding == 'lstm':
                        model.hidden_sentences = model.init_hidden(batch_size=sentences_emb.shape[1]) #(L,B,D) -> (109,8..,300)
                        #model.hidden = model.init_hidden(batch_size=int(sentences_emb.shape[1]/model.taille_context))
                        # Step 3. Run our forward pass.
                        sentences_emb_ = model.forward_sentence(sentences_emb, X_length) #(32,300)
                        to_packed_X = []
                        to_packed_Y = []
                        ref = ref.squeeze(0)
                        #print(sentences_emb_.shape[0], model.taille_context, sentences_emb_.shape[0] - model.taille_context, ref.size())
                        for i in range(sentences_emb_.shape[0] - model.taille_context + 1):
                            to_packed_X.append(torch.index_select(sentences_emb_, 0, torch.tensor(list(range(i,i+model.taille_context)), device=device)))
                            to_packed_Y.append(torch.index_select(ref, 0, torch.tensor([i+(int(model.taille_context/2)-1)], device=device)))
                        sentences_emb = torch.stack(to_packed_X).transpose(0,1) #(n,8,300) -> (8,n,300)
                        sentences_emb = sentences_emb.to(device)
                        ref = torch.stack(to_packed_Y).transpose(0,1) #(n,1) -> (1,n)
                    
                    model.hidden = model.init_hidden(batch_size=sentences_emb.shape[1])

                    prediction = model(sentences_emb) #(32,1)    #(1,32,4096) or (109,8*32,300) ?
                    prediction = torch.squeeze(prediction, 1)
                    ref = torch.squeeze(ref, 0)
                    if aa:
                        print(prediction, ref)
                        aa = False

                    loss = criterion(prediction, ref) #targets)
                    losses_dev_.append(loss.item())
                    if not best_loss_dev or loss < best_loss_dev:
                        torch.save(model.state_dict(), path_model+'model_best_'+str(epoch)+'.pth.tar')
                        best_loss_dev = loss
                        id_best_loss_dev = epoch
                    
                    Y_pred.append(np.asarray(prediction.detach().to('cpu')))
                    Y_ref.append(np.asarray(ref.to('cpu')))
            
            # Calculate the EER
            Y_ref = np.concatenate(Y_ref, axis=None)
            Y_pred = np.concatenate(Y_pred, axis=None)
            fpr, tpr, threshold = roc_curve(Y_ref, Y_pred, pos_label=1)
            fnr = 1 - tpr
            eer_threshold_dev = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
            eer_dev = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold (AUC = %0.2f)' % (auc(fpr, tpr)))
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic curve')
            plt.legend(loc="lower right")
            #plt.show()
            plt.savefig(config['path_work']+'roc_curve_'+str(epoch)+'.pdf')
            plt.close()
            
            mean_loss_dev = np.mean(np.asarray(losses_dev_))
            print('Mean loss on dev', mean_loss_dev)
            print('EER threshold, fpr (train/dev)', eer_threshold_train, eer_train, eer_threshold_dev, eer_dev)
            print('id_best_loss_dev', id_best_loss_dev)
            losses_dev.append(mean_loss_dev)
            #scheduler.step(mean_loss_dev)
            writer.add_scalars('data/scalar_group', {'loss_train': mean_loss_train, 'loss_dev': mean_loss_dev,
                                             'score_train': eer_train, 'score_dev': eer_dev}, epoch)
            writer.add_pr_curve('score_dev', np.mean(Y_ref), np.mean(Y_pred), epoch)
            #writer.add_pr_curve('roc_dev', np.mean(Y_ref), np.mean(Y_pred), epoch)

    writer.close()
    print('Best model on epoch', id_best_loss_dev) #Aller chercher à main et copier le bon modèle
    #torch.save(model.state_dict(), path_model+'model_best_'+str(id_best_loss_dev)+'.pth.tar')
    print('fin train')
    return losses

def get_predictions(config, model, subset, path_data, path_work, save=False):  
    model = model.eval()
    
    device = config['device']
    type_sentence_embedding = config['type_sentence_embedding']
    
    #ids_iter=list(range(len(inputs_embeddings_train)))
    #ids_iter=list(range(len(inputs_embeddings_train)))
    #shuffle(ids_iter)
    # See what the scores are after training
    with torch.no_grad():
        error_global = 0
        Y_positives_all = []
        Y_negatives_all = []
        Y_pred = []
        Y_ref = []
        nb_files = len(list(glob.glob(path_data+'/'+subset+'/'+'inputs_embeddings_*.pickle')))
        for it_ in tqdm(range(nb_files)):
            #print(it_+1,'on',nb_files,' prediction')
            
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
                if sentences_emb.shape[0] == 0: #TODO il y a des tensors vide, par exemple le 142ème en partant de 0
                    continue
                sentences_emb = sentences_emb.to(device)
                ref = ref.to(device)

                if type_sentence_embedding == 'lstm':
                    model.hidden_sentences = model.init_hidden(batch_size=sentences_emb.shape[1]) #(L,B,D) -> (109,8..,300)
                    #model.hidden = model.init_hidden(batch_size=int(sentences_emb.shape[1]/model.taille_context))
                    # Step 3. Run our forward pass.
                    sentences_emb_ = model.forward_sentence(sentences_emb) #(32,300)
                    to_packed_X = []
                    to_packed_Y = []
                    ref = ref.squeeze(0)
                    #print(sentences_emb_.shape[0], model.taille_context, sentences_emb_.shape[0] - model.taille_context, ref.size())
                    for i in range(sentences_emb_.shape[0] - model.taille_context + 1):
                        to_packed_X.append(torch.index_select(sentences_emb_, 0, torch.tensor(list(range(i,i+model.taille_context)), device=device)))
                        to_packed_Y.append(torch.index_select(ref, 0, torch.tensor([i+(int(model.taille_context/2)-1)], device=device)))
                    sentences_emb = torch.stack(to_packed_X).transpose(0,1) #(n,8,300) -> (8,n,300)
                    sentences_emb = sentences_emb.to(device)
                    ref = torch.stack(to_packed_Y).transpose(0,1) #(n,1) -> (1,n)
                model.hidden = model.init_hidden(batch_size=sentences_emb.shape[1])
                prediction = model(sentences_emb)#.item()
                prediction = torch.squeeze(prediction, 1)
                ref = torch.squeeze(ref, 0)
                #print(prediction, ref, abs(ref - prediction))
                Y_pred.append(np.asarray(prediction.detach().to('cpu')))
                Y_ref.append(np.asarray(ref.to('cpu')))
                for i, v in enumerate(ref):
                    if np.asarray(v) == 1:
                        Y_positives_all.append(np.asarray(prediction.detach().to('cpu')[i]))
                    elif np.asarray(v) == 0:
                        Y_negatives_all.append(np.asarray(prediction.detach().to('cpu')[i]))
    
    Y_ref = np.concatenate(Y_ref, axis=None)
    Y_pred = np.concatenate(Y_pred, axis=None)
    fpr, tpr, threshold = roc_curve(Y_ref, Y_pred, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    if save:
        Y_positives_all = np.asarray(Y_positives_all)
        Y_negatives_all = np.asarray(Y_negatives_all)
        np.save(path_work+'Y_positives_+'+subset+'.npy', Y_positives_all)
        np.save(path_work+'Y_negatives_+'+subset+'.npy', Y_negatives_all)
        print('mean score at 0.5 threshold',np.mean(Y_positives_all), np.mean(Y_negatives_all), np.mean(Y_positives_all) - np.mean(Y_negatives_all))
    print('EER threshold, fpr', err_threshold, eer)
    return eer, eer_threshold