import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import numpy as np

import random

import matplotlib.pyplot as plt
%matplotlib inline


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Vocab:
    def __init__(self, data):      
        self.char2idx = {'START' : 0, 'END' : 1,'PAD' : 2,'UNK' : 3, '\n' : 4}
        idx = 5
        
        for i in data:
            if i not in self.char2idx:
                self.char2idx[i] = idx
                idx += 1
        self.idx2char = dict((v,k) for k,v in self.char2idx.items())
            
    def tokenize(self, sequence):
        res = []
        for char in sequence:
            if char in self.char2idx.keys():
                res.append(self.char2idx[char])
            else:
                res.append(3)
        return res
      
    
    def detokenize(self, sequence):
        return ''.join([self.idx2char[idx] for idx in sequence])
    
    def __len__(self):
        return len(self.char2idx)
    
    
    
class TextDataset(Vocab):
    def __init__(self, data_path):
        super().__init__
        self.data = [line for line in open(data_path, errors='ignore')]
    
        self.vocab = Vocab(''.join(self.data))
        
        
    def __len__(self):
        return len(self.data)
    
    
    def getidxs(self, sequence):
        return self.vocab.tokenize(sequence)
    
    
    def getletters(self, sequence):
        return self.vocab.detokenize(sequence)
    
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = self.vocab.tokenize(sample)
    
        # паддинг до maxlen
        lens = [len(i) for i in self.data]
        maxlen = max(lens)
        temp = [2 for i in range(maxlen)]
        length = len(sample)
        temp[:length] = sample
        
        sample = torch.LongTensor([0] + temp) # START
        
        temp.insert(length, 1) # END
        target = torch.LongTensor(temp)
        
        return sample, target

class LM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, tie_weights):
        super().__init__()
        
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.LogSoftmax(dim=2)
        
        if tie_weights:
            # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
            # https://arxiv.org/abs/1608.05859
            assert hidden_dim == embedding_dim
            self.decoder.weight = self.encoder.weight

        self.hid_dim = hidden_dim
        self.num_lays = num_layers

    def forward(self, embedding, hidden):
        emb = self.drop(self.encoder(embedding))
        emb = torch.unsqueeze(emb, 0)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        res = decoded.view(output.size(0), output.size(1), decoded.size(1))
        res = self.activation(res)
        return res, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_lays, batch_size, self.hid_dim)

def generate_token(idx, hidden, t):
    start = torch.LongTensor([idx]).to(device)
    out, hidden = model(start, hidden)  
    token_probas = out.squeeze().div(t).exp().cpu()
    token = torch.multinomial(token_probas, 1)[0] 
    return token, hidden
     
def sample(num_lines, seed="", t=1.0):
    model.eval()
    idxs = [0] + dataset.getidxs(seed)
    
    hidden = model.init_hidden(1).to(device)
    
    result = seed[:]
    
    
    for idx in idxs:
        last_token, hidden = generate_token(idx, hidden, t)
    result += dataset.getletters([last_token.tolist()])
       
    i = 0
    while i < num_lines:
        
        token, hidden = generate_token(last_token, hidden, t)
        
        if token == 1 or token == 2 or token == 4:# END or PAD
            result += '\n'
            hidden = model.init_hidden(1).to(device)
            i += 1
            last_token = 0
        else:
            result += dataset.getletters([token.tolist()])
            last_token = token  
    return result

from IPython.core.display import display, HTML

def print_colored(sequence, intensities, delimeter=''):
    html = delimeter.join([
        f'<span style="background: rgb({255}, {255-x}, {255-x})">{c}</span>'
        for c, x in zip(sequence, intensities) 
    ])
    display(HTML(html))


def spellcheck(sequence):
    model.eval()
    
    vec_sequence = torch.LongTensor([dataset.getidxs(sequence)]).to(device)

    hidden = model.init_hidden(vec_sequence.shape[0]).to(device)
    spellcheck_losses = []  
    for i in range(vec_sequence.shape[1]):
        tempin = torch.unsqueeze(vec_sequence[0][i], 0)
        
        out, hidden = model(tempin, hidden)
        
        loss = criterion(out[0], torch.unsqueeze(vec_sequence[0, i], 0))

        spellcheck_losses.append(loss ** 2)

    
    print_colored(sequence, np.array(spellcheck_losses))

dataset = TextDataset('data.txt')
    
train_size = int(1 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

epochs = 1000
lr = 1e-4
batch_size = 32

train = DataLoader(train_set, batch_size=batch_size, shuffle=True)

model = LM(
    vocab_size = len(dataset.vocab),
    embedding_dim = 128,
    hidden_dim = 128,
    num_layers = 2,
    dropout = 0.2,
    tie_weights= True
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.NLLLoss()#nn.CrossEntropyLoss()
e = 0
evolution = {}

loss_history = []

while e < epochs:
    print(e)
    
    model.train()
    for x, y in train:
        
        x = x.to(device)
        y = y.to(device)
        model.to(device) 
        
        hidden = model.init_hidden(x.shape[0]).to(device) # x.shape[0] == batch_size; x.shape[1] == maxlen  
        
        for i in range(x.shape[1]):
        
            model.zero_grad()
            
            out, hidden = model(x[:, i], hidden)

            loss = criterion(out[0], y[:, i])
      
            loss.backward(retain_graph=True)
        
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)

            loss_history.append(loss.item())

            optimizer.step()
            
            
    samp = sample(3, t=random.uniform(0.5, 1))
    print(samp)
    evolution[e] = samp
    e += 1
