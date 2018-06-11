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
    return torch.tensor([token_to_id(t) for t in sent.tokens])


class LSTMLower(nn.Module):
    """ LSTM over a Batch of variable length sentences, maxpool over
    each sentence's hidden states to get its representation. """    
    def __init__(self, hidden_dim, num_layers, bidir):
        super().__init__()
        
        weights = VECTORS.weights()
        
        self.embeddings = nn.Embedding(weights.shape[0], weights.shape[1])
        self.embeddings.weight.data.copy_(weights)
        self.embeddings.requires_grad = False
        
        self.lstm = nn.LSTM(weights.shape[1], hidden_dim, num_layers=num_layers,
                            bidirectional=bidir, batch_first=True)

    def forward(self, batch):
        
        # Convert sentences to embed lookup ID tensors
        sent_tensors = [sent_to_tensor(s) for s in batch]
        
        # Embed tokens in each sentence
        embedded = [self.embeddings(s) for s in sent_tensors]
        
        # Pad, pack  embeddings of variable length sequences of tokens
        packed, reorder = pad_and_pack(embedded)
                
        # LSTM over the packed embeddings
        lstm_out, _ = self.lstm(packed)
                
        # Unpack, unpad, and restore original ordering of lstm outputs
        restored = unpack_and_unpad(lstm_out, reorder)
                
        # Maxpool over the hidden states
        maxpooled = [F.max_pool2d(sent, (sent.shape[1], 1)) 
                     for sent in restored]
                
        # Stack and squeeze to pass into a linear layer
        stacked = torch.stack(maxpooled).squeeze()
        
        # Regroup the document sentences for next pad_and_pack
        lower_output = batch.regroup(stacked)
        
        return lower_output


class LSTMHigher(nn.Module):
    """ LSTM over the sentence representations from LSTMLower """
    def __init__(self, input_dim, hidden_dim, num_layers, bidir):
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidir, batch_first=True)

    def forward(self, lower_output):
        
        # Pad, pack variable length sentence representations
        packed, reorder = pad_and_pack(lower_output)
        
        # LSTM over sentence representations
        lstm_out, _ = self.lstm(packed)
        
        # Restore original ordering of sentences
        restored = unpack_and_unpad(lstm_out, reorder)
        
        # Concatenate the sentences together for final scoring
        higher_output = torch.cat([sent.squeeze() for sent in restored])
        
        return higher_output


class Score(nn.Module):
    """ Take outputs from LSTMHigher, produce probabilities for each
    sentence that it ends a segment. """
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, out_dim),
        )
        
    def forward(self, higher_output):
        scores = self.score(higher_output)
        return scores


class TextSeg(nn.Module):
    """ Super class for taking an input batch of sentences from a Batch
    and computing the probability whether they end a segment or not """
    def __init__(self, lstm_dim, score_dim, bidir, num_layers=2):
        super().__init__()
        
        # Compute input dimension size for LSTMHigher, Score
        num_dirs = 2 if bidir else 1
        input_dim = lstm_dim*num_dirs
        
        # Chain modules together to get overall model
        self.model = nn.Sequential(
            LSTMLower(lstm_dim, num_layers, bidir),
            LSTMHigher(input_dim, lstm_dim, num_layers, bidir),
            Score(input_dim, score_dim, out_dim=2)
        )
        
    def forward(self, batch):
        return self.model(batch)


class Trainer:
    """ Class to train, validate, and test a model """
    def __init__(self, model, train_dir, val_dir, test_dir=None, 
                 lr=1e-3, batch_size=10):
        
        self.model = model
        self.optimizer = optim.Adam(params=[p for p in self.model.parameters() 
                                            if p.requires_grad], 
                                    lr=lr)
        
        self.best_val = 1e10
        self.batch_size = batch_size
        
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
    
    def train(self, num_epochs, *args, **kwargs):
        """ Train a model """
        for epoch in range(1, num_epochs+1):
            self.train_epoch(epoch, *args, **kwargs)
    
    def train_epoch(self, epoch, steps=25, val_ckpt=5):
        """ Train the model for one epoch """
        
        # Enable dropout, any learnable regularization
        self.model.train()
        
        epoch_loss, epoch_sents = [], []
        for step in tqdm_notebook(range(1, steps+1)):
            
            # Zero out gradients
            self.optimizer.zero_grad()
            
            # Compute a train batch, backpropagate
            batch_loss, num_sents, segs_correct, total_segs = self.train_batch()
            batch_loss.backward()
            
            print('Step: %d | Loss: %f | Num. sents: %d | Segs correct: %d / %d'
                 % (step, batch_loss.item(), num_sents, segs_correct, total_segs))
            
            # For logging purposes
            epoch_loss.append(batch_loss.item())
            epoch_sents.append(num_sents)
            
            # Step the optimizer
            self.optimizer.step()
        
        # Log progress
        print('Epoch: %d | Loss: %f | Avg num sents: %d\n' 
              % (epoch, np.mean(epoch_loss), np.mean(epoch_sents))) 
        
        # Validation set performance
        if val_ckpt % epoch:
            val_loss = self.validate()
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.best_model = deepcopy(self.model)
            
            # Log progress
            print('Validation loss: %f | Best val loss: %f\n' 
                  % (val_loss, self.best_val))

    def train_batch(self):
        """ Train the model using one batch """
        
        # Sample a batch of documents
        batch = sample_and_batch(self.train_dir, self.batch_size, TRAIN=True)

        # Get predictions for each document in the batch
        preds = self.model(batch)

        # Compute cross entropy loss, ignoring last entry as it always
        # ends a subsection without exception
        batch_loss = F.cross_entropy(preds, batch.labels, 
                                     size_average=False)
        
        print([(F.softmax(p, dim=0), l.item()) for p, l in zip(preds, batch.labels)])
        
        logits = F.softmax(preds, dim=1)
        probs, outputs = torch.max(logits, dim=1)
        
        segs_correct, total_segs = self.debugging(preds, batch)
        
        return batch_loss, len(batch), segs_correct, total_segs
    
    def debugging(self, preds, batch):
        """ Check how many segment boundaries were correctly predicted """
        labels = batch.labels
        logits = F.softmax(preds, dim=1)
        probs, outputs = torch.max(logits, dim=1)
        segs_correct = sum([1 for i,j in zip(batch.labels, outputs) 
                            if i == j == torch.tensor(1)])
        
        return segs_correct, sum(batch.labels).item()
        
    def validate(self):
        """ Compute performance of the model on a valiation set """
        
        # Disable dropout, any learnable regularization
        self.model.eval()
        
        # Retrieve all files in the val directory
        files = list(crawl_directory(self.val_dir))

        # Compute loss on this dataset
        val_loss = 0
        for chunk in chunk_list(files, self.batch_size):
            batch = Batch([read_document(f, TRAIN=False) for f in chunk])
            preds = self.model(batch)
            loss = F.cross_entropy(preds[:-1], batch.labels[:-1])
            val_loss += loss

        return val_loss.item()
    
    def predict(self, document):
        """ Given a document, predict segmentations """
        return self.predict_batch(Batch([document]))

    def predict_batch(self, batch, THETA=0.50):
        """ Given a batch, predict segmentation boundaries thresholded 
        by min probability THETA, which needs to be tuned """
        preds = model(batch)
        logits = F.softmax(preds, dim=1)
        outputs = logits[:, 1] > THETA
        boundaries = outputs.tolist()
        return boundaries

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath + '.pth')

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
        self.model = to_cuda(self.model)
