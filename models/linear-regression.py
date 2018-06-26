import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm_notebook, tqdm

from loader import *
from utils import *
from metrics import Metrics, avg_dicts


class LinearRegression(nn.Module):
    """ Linear regression baseline using discrete, binary bag of words
    """
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
    """ Featurize batch tokens into sparse vectors
    """
    
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
    """ Class to train, validate, and test a model 
    """
    
    # Progress logging, initialization of metrics
    train_loss = []
    val_loss = [] 
    metrics = defaultdict(list)
    best_val = 1e10
    evalu = Metrics()
    
    def __init__(self, model, train_dir, val_dir, test_dir=None, 
                 lr=1e-3, batch_size=10):
        
        self.model = model
        self.optimizer = optim.Adam(params=[p for p in self.model.parameters() 
                                            if p.requires_grad], 
                                    lr=lr)

        self.batch_size = batch_size
        
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir if test_dir else val_dir
            
    def train(self, num_epochs, *args, **kwargs):
        """ Train a model """
        for epoch in range(1, num_epochs+1):
            self.train_epoch(epoch, *args, **kwargs)
    
    def train_epoch(self, epoch, steps=25, val_ckpt=5, visualize=True):
        """ Train the model for one epoch """
        self.val_ckpt = val_ckpt
        
        # Enable dropout, any learnable regularization
        self.model.train()
        
        epoch_loss, epoch_sents = [], []
        for step in tqdm_notebook(range(1, steps+1)):
            
            # Zero out gradients
            self.optimizer.zero_grad()
            
            # Compute a train batch, backpropagate
            batch_loss, num_sents, segs_correct, total_segs = self.train_batch()
            batch_loss.backward()
            
            # Log progress (Loss is reported as average loss per sentence)
            print('Step: %d | Loss: %f | Num. sents: %d | Segs correct: %d / %d'
                 % (step, batch_loss.item()/num_sents, num_sents, segs_correct, total_segs))
            
            # For logging purposes
            epoch_loss.append(batch_loss.item())
            epoch_sents.append(num_sents)
            
            # Step the optimizer
            self.optimizer.step()
        
        epoch_loss = np.mean(epoch_loss)
        epoch_sents = np.mean(epoch_sents)
        
        # Log progress (Loss is reported as average loss per sentence)
        print('\nEpoch: %d | Loss: %f | Avg. num sents: %d\n' 
              % (epoch, epoch_loss/epoch_sents, epoch_sents))
        
        self.train_loss.append(epoch_loss/epoch_sents)
        
        # Validation set performance
        if epoch % val_ckpt == 0:
            
            metrics_dict, val_loss = self.validate(self.val_dir)
            
            # Log progress
            self.val_loss.append(val_loss)
            for key, val in metrics_dict.items():
                self.metrics[key].append(val)
                            
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.best_model = deepcopy(self.model.eval())
            
            # Log progress
            print('Validation loss: %f | Best val loss: %f\n' 
                  % (val_loss, self.best_val))
            
            if visualize:
                self.viz()

    def train_batch(self):
        """ Train the model using one batch """
        
        # Sample a batch of documents
        batch = sample_and_batch(self.train_dir, self.batch_size, TRAIN=True)

        # Get predictions for each document in the batch
        preds = self.model(batch)

        # Compute loss, IGNORING last entry as it ALWAYS ends a subsection
        batch_loss = F.cross_entropy(preds[:-1], batch.labels[:-1], size_average=False)
        
        # Number of boundaries correctly predicted
        segs_correct, total_segs = self.debugging(preds, batch)
        
        return batch_loss, len(batch), segs_correct, total_segs
    
    def debugging(self, preds, batch):
        """ Check how many segment boundaries were correctly predicted """
        labels = batch.labels
        logits = F.softmax(preds, dim=1)
        probs, outputs = torch.max(logits, dim=1)
        segs_correct = sum([1 for i,j in zip(batch.labels, outputs)
                            if i == j == torch.tensor(1)])
        total_segs = sum(batch.labels).item()
        
        print('\nBoundary Probabilities:\n')
        print([(logit[1].item(), label.item()) 
               for logit, label in zip(logits, batch.labels)
               if label == 1])
        
        return segs_correct, total_segs
    
    def validate(self, dirname):
        """ Evaluate using SegEval text segmentation metrics """
        
        print('Evaluating across SegEval metrics.')
        
        # Disable dropout, any learnable regularization
        self.model.eval()
        
        # Initialize val directory files, dictionaries list
        files, dicts_list = list(crawl_directory(dirname)), []
        
        eval_loss, num_sents = 0, 0
        # Break into chunks for memory constraints
        for chunk in chunk_list(files, self.batch_size):
            
            # Batchify documents
            batch = Batch([read_document(f, TRAIN=False) for f in chunk])
            
            # Predict the batch
            preds, logits = self.predict_batch(batch)
            
            # Compute validation loss, add number of sentences
            eval_loss += F.cross_entropy(logits, batch.labels, size_average=False)
            num_sents += len(batch)
            
            # Evaluate across SegEval metrics
            metric_dict = self.evalu(batch, preds)
            
            # Save the batch performance
            dicts_list.append(metric_dict)

        # Average dictionaries
        eval_metrics = avg_dicts(dicts_list)
        
        # Normalize eval loss
        normd_eval_loss = eval_loss.item() / num_sents
        
        return eval_metrics, normd_eval_loss
    
    def predict(self, document):
        """ Given a document, predict segmentations """
        return self.predict_batch(Batch([document]))

    def predict_batch(self, batch, THETA=0.50):
        """ Given a batch, predict segmentation boundaries thresholded 
        by min probability THETA, which needs to be tuned """
        
        # Predict
        logits = self.model(batch)
        
        # Softmax for probabilities
        probs = F.softmax(logits, dim=1)
        
        # If greater than threshold theta, make it a boundary
        boundaries = probs[:, 1] > THETA
        
        # Convert from tensor to list
        preds = boundaries.tolist()
        
        return preds, logits

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath + '.pth')

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
        self.model = to_cuda(self.model)
        
    def viz(self):
        """ Visualize progress: train loss, val loss, word- sent-level metrics """
        # Initialize plot
        _, axes = plt.subplots(ncols=2, nrows=2, sharex='col', sharey='col')
        val, word, train, sent = axes.ravel()

        # Plot validation loss
        val.plot(self.val_loss, c='g')
        val.set_ylabel('Val Loss')
        val.set_ylim([0,1])

        # Plot training loss
        train.plot(self.train_loss, c='r')
        train.set_ylabel('Train Loss')
        
        for key, values in self.metrics.items():
            
            # Plot word-level metrics
            if key.startswith('w_'):
                word.plot(values, label=key)
                
            # Plot sent-level metrics
            elif key.startswith('s_'):
                sent.plot(values, label=key)

        # Fix y axis limits, y label, legend for word-level metrics
        word.set_ylim([0,1])
        word.set_ylabel('Word metrics')
        word.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        
        # Fix again but this time for sent-level
        sent.set_ylabel('Sent metrics')
        sent.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

        # Give the plots some room to breathe
        plt.subplots_adjust(left=None, bottom=4, right=2, top=5,
                            wspace=None, hspace=None)

        # Display the plot
        plt.show()


model = LinearRegression()

trainer = Trainer(model=model,
                  train_dir='../data/wiki_727/train', 
                  val_dir='../data/wiki_50/test',
                  test_dir=None,
                  batch_size=32,
                  lr=1e-3)

trainer.train(num_epochs=100, 
              steps=100)

