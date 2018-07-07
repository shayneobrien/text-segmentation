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
    def __init__(self, hidden_dim, num_layers, bidir, drop_prob, method):
        super().__init__()

        weights = VECTORS.weights()

        self.embeddings = nn.Embedding(weights.shape[0], weights.shape[1])
        self.embeddings.weight.data.copy_(weights)

        self.drop = drop_prob

        self.lstm = nn.LSTM(weights.shape[1], hidden_dim,
                            num_layers=num_layers, bidirectional=bidir,
                            batch_first=True, dropout=self.drop)

        self.pool = Pool(method)

    def forward(self, batch):

        # Convert sentences to embed lookup ID tensors
        sent_tensors = [sent_to_tensor(s) for s in batch]

        # Embed tokens in each sentence
        embedded = [F.dropout(self.embeddings(s), self.drop) for s in sent_tensors]

        # Pad, pack  embeddings of variable length sequences of tokens
        packed, reorder = pad_and_pack(embedded)

        # LSTM over the packed embeddings
        lstm_out, _ = self.lstm(packed)

        # Unpack, unpad, and restore original ordering of lstm outputs
        restored = unpack_and_unpad(lstm_out, reorder)

        # Get lower output representation
        representation = self.pool(restored)

        # Regroup the document sentences for next pad_and_pack
        lower_output = torch.stack(representation)

        return lower_output


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


class Pool:
    """ Given regrouped representations from batch, perform pooling over them
    using one of several methods.
    """
    def __init__(self, method):
        assert method in ['avg', 'last', 'max', 'sum', 'weighted'], 'Invalid method chosen.'
        self.method = eval('self._' + method)

    def __call__(self, *args):
        return self.method(*args)

    def _weighted(self, restored):
        """ Weighted sum  """
        weighted = [F.softmax(sent, dim=0)*sent for sent in restored]
        return self._sum(weighted)

    def _avg(self, restored):
        """ Average hidden states """
        return [sent.mean(dim=0) for sent in restored]

    def _last(self, restored):
        """ Take last token state representation """
        return [sent[-1, :] for sent in restored]

    def _max(self, restored):
        """ Maxpool token states """
        return [torch.max(sent, dim=0)[0] for sent in restored]

    def _sum(self, restored):
        """ Sum token states """
        return [sent.sum(dim=0) for sent in restored]


class LSTMScore(nn.Module):
    """ Super class for taking an input batch of sentences from a Batch
    and computing the probability whether they end a segment or not
    """
    def __init__(self, lstm_dim, score_dim,
                       bidir, num_layers=2,
                       drop_prob=0.20, method='max'):
        super().__init__()

        # Compute input dimension size for LSTMHigher, Score
        num_dirs = 2 if bidir else 1
        input_dim = lstm_dim*num_dirs

        # Chain modules together to get overall model
        self.model = nn.Sequential(
            LSTMLower(lstm_dim, num_layers, bidir, drop_prob, method),
            Score(input_dim, score_dim, out_dim=2, drop_prob=drop_prob)
        )

    def forward(self, batch):
        return self.model(batch)


if __name__ == '__main__':
    model = LSTMScore(lstm_dim=256,
                      score_dim=256,
                      bidir=True,
                      num_layers=2,
                      drop_prob=0.20,
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
