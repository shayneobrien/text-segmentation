import os, sys
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.trainer import *


class CNNEncoder(nn.Module):
    """ CNN over a Batch of variable length sentences padded
    or truncated to maxlen. maxpool over
    each sentence's hidden states to get its representation.
    """
    def __init__(self, hidden_dim, drop_prob, maxlen, nrange, method):
        super().__init__()

        assert type(nrange) == list, 'Argument "nrange" is a list of token convolution sizes.'
        assert method in ['avg', 'last', 'max', 'sum'], 'Invalid method chosen.'
        self.method = eval('self._'+ method)

        weights = VECTORS.weights()

        self.embeddings = nn.Embedding(weights.shape[0], weights.shape[1], padding_idx=0)
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
        embedded = F.dropout(self.embeddings(padded), 0.20).squeeze().transpose(1,2)

        # Convolve over words
        convolved = [conv(embedded) for conv in self.convs]

        # Cat together convolutions
        catted = torch.cat(convolved, dim=2)

        # Squash down a dimension
        representation = self.method(catted)

        return representation

    def _avg(self, catted):
        """ Average token states """
        return catted.mean(dim=1)

    def _max(self, catted):
        """ Maxpool token states """
        return F.max_pool2d(catted, kernel_size=(catted.shape[1], 1)).squeeze()

    def _sum(self, catted):
        """ Sum token states """
        return catted.sum(dim=1)


class Score(nn.Module):
    """ Take outputs from encoder, produce probabilities for each
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


class CNNScore(nn.Module):
    """ Super class for taking an input batch of sentences from a Batch
    and computing the probability whether they end a segment or not
    """
    def __init__(self, hidden_dim, score_dim, drop_prob, maxlen, nrange, method):
        super().__init__()

        # Compute input dimension size for Score
        input_dim = sum([maxlen - (n-1) for n in nrange])

        # Chain modules together to get overall model
        self.model = nn.Sequential(
            CNNEncoder(hidden_dim, drop_prob, maxlen, nrange, method),
            Score(input_dim, score_dim, out_dim=2, drop_prob=drop_prob)
        )

    def forward(self, batch):
        return self.model(batch)


if __name__ == '__main__':
    model = CNNScore(hidden_dim=256,
                     score_dim=256,
                     drop_prob=0.20,
                     maxlen=64,
                     nrange=[3,4,5],
                     method='max')

    trainer = Trainer(model=model,
                      train_dir='../data/wiki_727/train',
                      val_dir='../data/wiki_50/test',
                      batch_size=8,
                      lr=1e-3,
                      visualize=False)

    trainer.train(num_epochs=100,
                  steps=25,
                  val_ckpt=1)
