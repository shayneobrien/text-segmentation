import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
from tqdm import tqdm_notebook, tqdm

from loader import *
from utils import *


class LinearRegression(nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()
        
        self.embeddings = BagOfWords()
        self.linear = nn.Linear(self.embeddings.vocab_size, output_dim)
    
    def forward(self, batch):
        embedded = self.embeddings(batch)
        activated = self.linear(embedded)
        preds = F.sigmoid(activated).squeeze()
        return preds
    

class BagOfWords:
    
    unk_idx = 0
    
    def __init__(self):
        self.set_vocab()
        
    def __call__(self, *args):
        return self.featurize(*args)
    
    def featurize(self, batch):
        """ Turn batch or document into discrete, binarized bag of words """
        sentences = [s.tokens for s in batch.sents]
        token_idx = [[self.stoi(t) for t in tokenized] 
                      for tokenized in sentences]
        
        featurized = torch.zeros((len(token_idx), self.vocab_size))
        for sent_idx, counts in enumerate(token_idx):
            featurized[sent_idx, counts] += 1
        
        return to_var(featurized)
    
    def set_vocab(self):
        """Set corpus vocab """
        # Map string to intersected index.
        self._stoi = {s: i for i, s in enumerate(self.get_vocab())}
        
        # Vocab size, plus one for UNK tokens (idx: 0)
        self.vocab_size = len(self._stoi.keys()) + 1 
        
    def get_vocab(self, filename='vocabulary.txt'):
        """ Read in vocabulary (top 30K words, covers ~93.5% of all tokens) """ 
        with open(filename, 'r') as f:
            vocab = f.read().split(',')
        return vocab
        
    def stoi(self, s):
        """ String to index (s to i) for embedding lookup """
        idx = self._stoi.get(s)
        return idx + 1 if idx else self.unk_idx
    

class Trainer:
    """ Trainer class for Linear Regression """    
    def __init__(self, model, batch_size, train_dir,
                val_dir, test_dir, lr=1e-3):
        
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
    
    def train_epoch(self, epoch, steps=100, val_ckpt=5):
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
        print('Epoch: %d | Loss: %f | Avg. num sents: %d\n' 
              % (epoch, np.mean(epoch_loss), np.mean(epoch_sents))) 
        
        # Validation set performance
        if val_ckpt % epoch:
            val_loss = self.validate(self.val_dir)
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
        
        print([(F.softmax(p, dim=0), l.item()) for p, l in zip(preds, batch.labels)][:10])
        
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
        
    def validate(self, dirname):
        """ Compute performance of the model on a valiation set """
        
        # Disable dropout, any learnable regularization
        self.model.eval()
        
        # Retrieve all files in the val directory
        files = list(crawl_directory(dirname))

        # Compute loss on this dataset
        val_loss = 0
        for chunk in chunk_list(files, self.batch_size):
            batch = Batch([read_document(f, TRAIN=False) for f in chunk])
            preds = self.model(batch)
            loss = F.cross_entropy(preds, batch.labels)
            val_loss += loss

        return val_loss.item()


model = LinearRegression()
trainer = Trainer(model=model,
                  train_dir='../data/wiki_727/train', 
                  val_dir='../data/wiki_50/test',
                  test_dir=None,
                  batch_size=32,
                  lr=1e-3)

trainer.train(num_epochs=100, steps=100)