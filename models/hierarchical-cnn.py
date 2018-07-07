import os, sys
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.trainer import *


class CNNLower(nn.Module):
    """ Convolve over words in windows within nrange for
    each sentence of a batch. Consider up to maxlen tokens.
    """
    def __init__(self, hidden_dim, drop_prob, maxlen, nrange):
        super().__init__()

        assert type(nrange) == list, 'Argument "nrange" should be list.'

        weights = VECTORS.weights()

        self.embeddings = nn.Embedding(weights.shape[0],
                                       weights.shape[1],
                                       padding_idx=0)
        self.embeddings.weight.data.copy_(weights)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=weights.shape[1],
                                              out_channels=hidden_dim,
                                              kernel_size=n)
                                    for n in nrange])

        self.drop = drop_prob
        self.maxlen = maxlen

    def forward(self, batch):

        # Convert sentences to embed lookup ID tensors
        sent_tensors = [sent_to_tensor(s).unsqueeze(1) for s in batch]

        # Pad to maximum length sentence in the batch
        padded, _ = pad_and_stack(sent_tensors, pad_size=self.maxlen)

        # Embed tokens in each sentence, apply dropout, transpose for input to CNN
        embedded = F.dropout(self.embeddings(padded), self.drop).squeeze().transpose(1,2)

        # Convolve over words
        convolved = [conv(embedded) for conv in self.convs]

        # Cat together convolutions
        catted = torch.cat(convolved, dim=2)

        # Regroup into document boundaries
        lower_output = batch.regroup(catted)

        return lower_output


class CNNHigher(nn.Module):
    """ Convolve over sentence representations from each document
    """
    def __init__(self, input_dim, hidden_dim, drop_prob, nrange, method):
        super().__init__()

        assert type(nrange) == list, 'Argument "nrange" should be list.'
        assert method in ['avg', 'last', 'max', 'sum'], 'Invalid method.'

        self.method = eval('self._'+ method)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=input_dim,
                                              out_channels=hidden_dim,
                                              kernel_size=n)
                                    for n in nrange])

        self.drop = drop_prob

    def forward(self, lower_output):

        # Convolve over sentences in each document
        convolved = [[conv(t) for conv in self.convs
                      if t.shape[2] >= conv.kernel_size[0]]
                     for t in lower_output]

        # Cat the convolutions together
        catted = [torch.cat(t, dim=2) for t in convolved]

        # Squash down dimension
        higher_output = torch.cat([self.method(t) for t in catted], dim=0)

        return higher_output

    def _avg(self, catted):
        """ Average sentence states """
        return catted.mean(dim=1)

    def _max(self, catted):
        """ Maxpool over sentence states """
        return F.max_pool2d(catted, kernel_size=(catted.shape[1], 1)).squeeze()

    def _sum(self, catted):
        """ Sum sentence states """
        return catted.sum(dim=1)


class Score(nn.Module):
    """ Take outputs from CNNHigher, produce probabilities for each
    sentence that it ends a segment.
    """
    def __init__(self, input_dim, hidden_dim, out_dim, drop_prob):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, higher_output):
        return self.score(higher_output)


class HierarchicalCNN(nn.Module):
    """ Super class for taking an input batch of sentences from a Batch
    and computing the probability whether they end a segment or not
    """
    def __init__(self, hidden_dim, score_dim, drop_prob, maxlen,
                 low_nrange, high_nrange, method):
        super().__init__()

        # Compute input dimension size for Score
        lower_dim = sum([maxlen - (n-1) for n in low_nrange])
        input_dim = sum([lower_dim - (n-1) for n in high_nrange])

        # Chain modules together to get overall model
        self.model = nn.Sequential(
            CNNLower(hidden_dim, drop_prob, maxlen, low_nrange),
            CNNHigher(hidden_dim, hidden_dim, drop_prob, high_nrange, method),
            Score(input_dim, score_dim, out_dim=2, drop_prob=drop_prob)
        )

    def forward(self, batch):
        return self.model(batch)


if __name__ == '__main__':
    model = HierarchicalCNN(hidden_dim=256,
                            score_dim=256,
                            drop_prob=0.20,
                            maxlen=64,
                            low_nrange=[3,4,5],
                            high_nrange=[1,2,3],
                            method='max')

    trainer = Trainer(model=model,
                      train_dir='../data/wiki_727/train',
                      val_dir='../data/wiki_50/test',
                      batch_size=4,
                      lr=1e-3,
                      visualize=False)

    trainer.train(num_epochs=100,
                  steps=5,
                  val_ckpt=1)
