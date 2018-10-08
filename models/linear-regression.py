import os, sys
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.trainer import *


class LinearRegression(nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()

        self.embeddings = BagOfWords()
        self.linear = nn.Linear(self.embeddings.vocab_size, output_dim)

    def forward(self, batch):
        embedded = self.embeddings(batch)
        preds = self.linear(embedded)
        # Sigmoid?
        return preds.squeeze()


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

    def get_vocab(self, filename='../src/vocabulary.txt'):
        """ Read in vocabulary (top 30K words, covers ~93.5% of all tokens) """
        with open(filename, 'r') as f:
            vocab = f.read().split(',')
        return vocab

    def stoi(self, s):
        """ String to index (s to i) for embedding lookup """
        idx = self._stoi.get(s)
        return idx + 1 if idx else self.unk_idx


if __name__ == '__main__':
    model = LinearRegression()

    trainer = Trainer(model=model,
                      train_dir='../data/wiki_727/train',
                      val_dir='../data/wiki_50/test',
                      test_dir=None,
                      batch_size=256,
                      lr=5e-4,
                      visualize=False)

    trainer.train(num_epochs=100,
                  steps=100,
                  val_ckpt=1)
