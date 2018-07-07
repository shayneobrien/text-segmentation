import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

from src.loader import *
from src.utils import *
from src.metrics import Metrics, avg_dicts


# Load in corpus, lazily load in word vectors.
VECTORS = LazyVectors()

def token_to_id(token):
    """ Lookup word ID for a token """
    return VECTORS.stoi(token)

def sent_to_tensor(sent):
    """ Convert a sentence to a lookup ID tensor """
    idx_tensor = torch.tensor([token_to_id(t) for t in sent.tokens])
    if idx_tensor.shape[0] == 0: # Empty, edge case; return UNK
        return torch.tensor([VECTORS.unk_idx])
    else:
        return idx_tensor


class Trainer:
    """ Class to train, validate, and test a model """

    # Progress logging, initialization of metrics
    train_loss = []
    val_loss = []
    metrics = defaultdict(list)
    best_val = 1e10
    evalu = Metrics()

    def __init__(self, model, train_dir, val_dir, test_dir=None,
                 lr=1e-3, batch_size=10, visualize=True):

        self.__dict__.update(locals())
        self.optimizer = optim.Adam(params=[p for p in self.model.parameters()
                                            if p.requires_grad],
                                    lr=self.lr)

    def train(self, num_epochs, *args, **kwargs):
        """ Train a model """
        for epoch in range(1, num_epochs+1):
            self.train_epoch(epoch, *args, **kwargs)

    def train_epoch(self, epoch, steps=25, val_ckpt=5):
        """ Train the model for one epoch """
        self.val_ckpt = val_ckpt

        # Enable dropout, any learnable regularization
        self.model.train()

        epoch_loss, epoch_sents = [], []
        for step in tqdm(range(1, steps+1)):

            # Zero out gradients
            self.optimizer.zero_grad()

            # Compute a train batch, backpropagate
            batch_loss, num_sents, segs_correct, texts_correct, total_segs, total_texts = self.train_batch()
            batch_loss.backward()

            # Log progress (Loss is reported as average loss per sentence)
            print('Step: %d | Loss: %f | Num. sents: %d | Segs correct: %d / %d | Texts correct: %d / %d'
                 % (step, batch_loss.item()/num_sents, num_sents, segs_correct, total_segs, texts_correct, total_texts))

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

            if self.visualize:
                self.viz_metrics()

    def train_batch(self):
        """ Train the model using one batch """

        # Sample a batch of documents
        batch = sample_and_batch(self.train_dir, self.batch_size, TRAIN=True)

        # Get predictions for each document in the batch
        preds = self.model(batch)

        # Compute loss, IGNORING last entry as it ALWAYS ends a subsection
        batch_loss = F.cross_entropy(preds[:-1], batch.labels[:-1],
                                     size_average=False, weight=self.weights(batch))

        # Number of boundaries correctly predicted
        segs_correct, texts_correct, total_segs, total_texts = self.debugging(preds, batch)

        return batch_loss, len(batch), segs_correct, texts_correct, total_segs, total_texts

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
            eval_loss += F.cross_entropy(logits, batch.labels,
                                         size_average=False, weight=self.weights(batch))
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

    def viz_metrics(self):
        """ Visualize progress: train loss, val loss, word- sent-level metrics """
        # Initialize plot
        _, axes = plt.subplots(ncols=2, nrows=2, sharex='col', sharey='col')
        val, word, train, sent = axes.ravel()

        # Plot validation loss
        val.plot(self.val_loss, c='g')
        val.set_ylabel('Val Loss')
        val.set_ylim([0,max(max(self.val_loss), max(self.train_loss))+0.1])

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

    def debugging(self, preds, batch, show_probs=True):
        """ Check how many segment boundaries were correctly predicted """
        labels = batch.labels
        logits = F.softmax(preds, dim=1)
        probs, outputs = torch.max(logits, dim=1)
        segs_correct = sum([1 for i,j in zip(batch.labels, outputs)
                            if i == j == torch.tensor(1)])

        texts_correct = sum([1 for i,j in zip(batch.labels, outputs)
                             if i == j == torch.tensor(0)])

        total_segs = batch.labels.sum().item()

        total_texts = (batch.labels == 0).sum().item()

        if show_probs:
            means = logits.mean(dim=0)
            print('Label 0: %f | Label 1: %f'
                   % (means[0].item(), means[1].item()))

        return segs_correct, texts_correct, total_segs, total_texts

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

    def weights(self, batch):
        """ Class weight loss from batch """
        zero_weight = 1/(len(batch.labels) / sum(batch.labels).float())
        one_weight = torch.tensor(1.)
        return torch.stack([zero_weight, one_weight])

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath + '.pth')

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
        self.model = to_cuda(self.model)
