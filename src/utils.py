import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

def to_var(x):
    """ Convert a tensor to a backpropable tensor """
    return to_cuda(x).requires_grad_()

def to_cuda(x):
    """ Cuda-erize a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def unpack_and_unpad(lstm_out, reorder):
    """ Given a padded and packed sequence and its reordering indexes,
    unpack and unpad it. Inverse of pad_and_pack """

    # Restore a packed sequence to its padded version
    unpacked, sizes = pad_packed_sequence(lstm_out, batch_first=True)

    # Restored a packed sequence to its original, unequal sized tensors
    unpadded = [unpacked[idx][:val] for idx, val in enumerate(sizes)]

    # Restore original ordering
    regrouped = [unpadded[idx] for idx in reorder]

    return regrouped

def pad_and_pack(sentences):
    """ Pad and pack a list of sentences to allow for batching of
    variable length sequences in LSTM modules """

    # Pad variables to longest observed length, stack them
    padded, sizes = pad_and_stack(sentences)

    # Pack the variables to mask the padding
    return pack(padded, sizes)

def pad_and_stack(tensors, pad_size=None):
    """ Pad and stack an uneven tensor of token lookup ids.
    Assumes num_sents in first dimension (batch_first=True)"""

    # Get their original sizes (measured in number of tokens)
    sizes = [s.shape[0] for s in tensors]

    # Pad size will be the max of the sizes
    if not pad_size:
        pad_size = max(sizes)

    # Pad all sentences to the max observed size
    padded = torch.stack([F.pad(sent[:pad_size], (0, 0, 0, max(0, pad_size-size)))
                          for sent, size in zip(tensors, sizes)])

    return padded, sizes

def pack(padded, sizes, batch_first=True):
    """Pack padded variables, provide reorder indexes """

    # Get indexes for sorted sizes (largest to smallest)
    size_sort = np.argsort(sizes)[::-1].tolist()

    # Resort the tensor accordingly
    padded = padded[torch.tensor(size_sort, requires_grad=False)]

    # Resort sizes in descending order
    sizes = sorted(sizes, reverse=True)

    # Pack the padded sequences
    packed = pack_padded_sequence(padded, sizes, batch_first)

    # Regroup indexes for restoring tensor to its original order
    reorder = torch.tensor(np.argsort(size_sort), requires_grad=False)

    return packed, reorder
