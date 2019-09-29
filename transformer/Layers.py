import torch
import torch.nn.functional as F
import numpy as np


def subsequent_mask(length):
    """ Returns a mask such that for position i, all positions i+1 ... dim are masked. """
    shape = (1, length, length)
    mask = 1 - np.triu(np.ones(shape), k=1)
    return torch.from_numpy(mask).bool()


class SublayerConnection(torch.nn.Module):
    """ Does residual + layer norm of input. Modular design inspired from Harvard NLP.
        http://nlp.seas.harvard.edu/2018/04/03/attention.html#encoder-and-decoder-stacks
    """
    def __init__(self, size):
        super(SublayerConnection, self).__init__()
        self.norm = torch.nn.LayerNorm(size)

    def forward(self, layer_input, layer):
        return layer_input + layer(self.norm(layer_input))


class PositionwiseFeedForward(torch.nn.Module):
    """ Position-wise Feed Forward network sublayer for the Transformer model. """
    def __init__(self):
        super(PositionwiseFeedForward, self).__init__()

class PositionalEncoding(torch.nn.Module):
    """ Positional encoding layer for the Transformer model. """
    def __init__(self):
        super(PositionalEncoding, self).__init__()
