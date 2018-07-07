import os, sys
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.trainer import *


class LSTMLower(nn.Module):
    """ LSTM over a Batch of variable length sentences, pool over
    each sentence's hidden states to get its representation.
    """
    def __init__(self, hidden_dim, num_layers, bidir, drop_prob, attn_dim):
        super().__init__()

        weights = VECTORS.weights()

        self.embeddings = nn.Embedding(weights.shape[0], weights.shape[1])
        self.embeddings.weight.data.copy_(weights)

        self.drop = drop_prob

        self.lstm = nn.LSTM(weights.shape[1], hidden_dim,
                            num_layers=num_layers, bidirectional=bidir,
                            batch_first=True, dropout=self.drop)

        self.attn = SelfAttention(attn_dim)

    def forward(self, batch):

        # Convert sentences to embed lookup ID tensors
        sent_tensors = [sent_to_tensor(s) for s in batch]

        # Embed tokens in each sentence
        embedded = [F.dropout(self.embeddings(s), self.drop) for s in sent_tensors]

        # Pad, pack  embeddings of variable length sequences of tokens
        packed, reorder = pad_and_pack(embedded)

        # LSTM over the packed embeddings
        lstm_out, _ = self.lstm(packed)

        # Compute attention over hidden states
        weighted = self.attn(lstm_out)

        # Restore original ordering
        lower_output = weighted[reorder]

        return lower_output


class Score(nn.Module):
    """ Take outputs from LSTMHigher, produce probabilities for each
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

    def forward(self, lower_output):
        return self.score(lower_output)


class SelfAttention(nn.Module):
    """ Module for computing dot self-attention
    """
    def __init__(self, attn_dim):
        super().__init__()

        # Initialize parameter and weights
        self.attn_weights = nn.Parameter(torch.Tensor(attn_dim))
        nn.init.uniform_(self.attn_weights.data, -0.005, 0.005)

        # [Viswani et al., 2017] to make dot attention approx. equal to
        # additive attention for larger hidden dim sizes
        self.scale = 1 / np.sqrt(attn_dim)

    def forward(self, lstm_out):
        """ Returns a singular unordered, padded 3D tensor. """
        # Unpack packed sequence
        unpacked, sizes = pad_packed_sequence(lstm_out, batch_first=True)

        # Dot the padded unpacked sequence with attention weights, activate
        activated = F.tanh(torch.matmul(unpacked, self.attn_weights)) * self.scale

        # Softmax over the activated tensor
        raw_scores = F.softmax(activated, dim=1)

        # Mask the portions of the sentence that should be padding
        mask = self._mask(sizes)

        # Mask padding
        masked_scores = raw_scores * mask

        # Renormalize softmax weights to ignore padding
        weights = self._normalize(masked_scores)

        # Weight hidden states
        scores = torch.mul(unpacked, weights.unsqueeze(-1).expand_as(unpacked))

        # Sum over tokens
        context = torch.sum(scores, dim=1)

        return context

    def _mask(self, sizes):
        """ Construct mask for padded itemsteps, based on lengths """
        pad_size = torch.max(sizes).item()
        mask = torch.stack([F.pad(torch.ones(int(size)), (0, pad_size-int(size)))
                     for size in sizes], dim=0)
        return mask

    def _normalize(self, masked_scores):
        """ Renormalize masked scores to ignore padding """
        sums = torch.sum(masked_scores, dim=1, keepdim=True)  # sums per row
        weights = torch.div(masked_scores, sums)  # divide by row sum
        return weights


class LSTMScore(nn.Module):
    """ Super class for taking an input batch of sentences from a Batch
    and computing the probability whether they end a segment or not
    """
    def __init__(self, lstm_dim, score_dim, bidir, num_layers=2, drop_prob=0.20):
        super().__init__()

        # Compute input dimension size for LSTMHigher, Score
        num_dirs = 2 if bidir else 1
        input_dim = lstm_dim*num_dirs

        # Chain modules together to get overall model
        self.model = nn.Sequential(
            LSTMLower(lstm_dim, num_layers, bidir, drop_prob, input_dim),
            Score(input_dim, score_dim, out_dim=2, drop_prob=drop_prob)
        )

    def forward(self, batch):
        return self.model(batch)


if __name__ == '__main__':
    model = LSTMScore(lstm_dim=256,
                      score_dim=256,
                      bidir=True,
                      num_layers=2,
                      drop_prob=0.20)

    trainer = Trainer(model=model,
                      train_dir='../data/wiki_727/train',
                      val_dir='../data/wiki_50/test',
                      batch_size=8,
                      lr=1e-3,
                      visualize=False)

    trainer.train(num_epochs=100,
                  steps=25,
                  val_ckpt=1)
