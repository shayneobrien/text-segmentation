import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm, tqdm_notebook

from loader import LazyVectors, sample_and_read
from utils import *

# Load in corpus, lazily load in word vectors.
VECTORS = LazyVectors()


def token_to_id(token):
    """ Lookup word ID for a token """
    return VECTORS.stoi(token)

def sent_to_tensor(sent):
    """ Convert a sentence to a lookup ID tensor """
    return to_var(torch.tensor([token_to_id(t) for t in sent.tokens]))


class LSTMLower(nn.Module):
    """ LSTM over a Batch of variable length sentences, maxpool over
    each sentence's hidden states to get its representation. """    
    def __init__(self, hidden_dim, num_layers, bidir):
        super().__init__()
        
        weights = VECTORS.weights()
        
        self.embeddings = nn.Embedding(weights.shape[0], weights.shape[1])
        self.embeddings.weight.data.copy_(weights)
        
        self.lstm = nn.LSTM(weights.shape[1], hidden_dim, num_layers=num_layers,
                            bidirectional=bidir, batch_first=True)

    def forward(self, batch):
        
        # Convert sentences to embed lookup ID tensors
        sent_tensors = [sent_to_tensor(s) for s in batch]
        
        # Embed tokens in each sentence
        embedded = [self.embeddings(s) for s in sent_tensors]
        
        # Pad and pack embeddings to deal with variable length sequences
        packed, reorder = pad_and_pack(embedded)
        
        # LSTM over the packed embeddings
        lstm_out, _ = self.lstm(packed)
                
        # Unpack, unpad, and restore original ordering of lstm outputs
        regrouped = unpack_and_unpad(lstm_out, reorder)
        
        # Maxpool over the hidden states
        maxpooled = [F.max_pool2d(sent, (sent.shape[1], 1)) 
                     for sent in regrouped]
        
        # Stack and squeeze to pass into a linear layer
        lower_output = torch.stack(maxpooled)
        
        return lower_output.squeeze()


class LSTMHigher(nn.Module):
    """ LSTM over the sentence representations from LSTMLower """
    def __init__(self, input_dim, hidden_dim, num_layers, bidir):
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidir, batch_first=True)

    def forward(self, lower_output):
        
        # LSTM needs 3D tensors, so add a dimension
        lower_output = lower_output.unsqueeze(0)
        
        # LSTM over sentence representations
        higher_output, _ = self.lstm(lower_output)
        
        return higher_output.squeeze()


class Score(nn.Module):
    """ Take outputs from LSTMHigher, produce probabilities for each
    sentence that it ends a segment. """
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, out_dim),
        )
        
    def forward(self, higher_output):
        return self.linear(higher_output).squeeze()


class TextSeg(nn.Module):
    """ Super class for taking an input batch of sentences from a Batch
    and computing the probability whether they end a segment or not """
    def __init__(self, lstm_dim, score_dim, bidir, num_layers=2):
        super().__init__()
        
        # Compute input dimension size for LSTMHigher, Score
        num_dirs = 2 if bidir else 1
        input_dim = lstm_dim*num_dirs
        
        # Chain
        self.segment = nn.Sequential(
            LSTMLower(lstm_dim, num_layers, bidir),
            LSTMHigher(input_dim, lstm_dim, num_layers, bidir),
            Score(input_dim, score_dim, out_dim=2)
        )
        
    def forward(self, batch):
        return self.segment(batch)
    
class Trainer:
    """ Class to train, validate, and test a model """
    def __init__(self, model, train_dir, val_dir=None, test_dir=None, 
                 lr=1e-3, batch_size=100):
        
        self.model = to_cuda(model)
        self.optimizer = optim.Adam(params=[p for p in self.model.parameters() if p.requires_grad], lr=lr)
        
        self.batch_size = batch_size
        
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
    
    def train(self, num_epochs, *args, **kwargs):
        """ Train a model """
        for epoch in range(1, num_epochs+1):
            self.train_epoch(epoch, *args, **kwargs)
    
    def train_epoch(self, epoch, steps=25):
        """ Train the model for one epoch """
        
        epoch_loss, epoch_sents = [], []
        for _ in tqdm_notebook(range(1, steps+1)):
            
            # Zero out gradients
            self.optimizer.zero_grad()
            
            # Compute a train batch, backpropagate
            batch_loss, num_sents = self.train_batch()
            batch_loss.backward()
            
            # For logging purposes
            epoch_loss.append(batch_loss.item())
            epoch_sents.append(num_sents)
            
            # Step the optimizer
            self.optimizer.step()

        print('Epoch: %d | Loss: %f | Avg num sents: %d' % (epoch, 
                                                            np.mean(epoch_loss)), 
                                                            np.mean(epoch_sents))

    def train_batch(self):
        """ Train the model for one batch """
        
        # Enable dropout, any regularization used
        self.model.train()
        
        # Sample a batch of documents
        batch = sample_and_read(self.train_dir, self.batch_size)
        
        batch_loss = 0
        for doc in tqdm_notebook(batch):
            
            # Get predictions for each document in the batch
            preds, labels = self.model(doc), to_var(torch.LongTensor(doc.labels))
            
            # Compute loss, aggregate
            loss = F.cross_entropy(preds, labels)
            batch_loss += loss
        
        # Compute number of sentences in the batch
        num_sents = sum([len(d) for d in batch])
        
        return batch_loss, num_sents
        
        
    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath + '.pth')

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
        self.model = to_cuda(self.model)