import torch
import torch.nn.functional as F
import numpy as np


def subsequent_mask(length):
    """ Returns a mask such that for position i, all positions i+1 ... dim are masked. """
    shape = (1, length, length)
    mask = 1 - np.triu(np.ones(shape), k=1)
    return torch.from_numpy(mask).bool()


class SublayerConnection(torch.nn.Module):
    """ Does residual + layer norm of input. Modular design inspired by Harvard NLP.
        http://nlp.seas.harvard.edu/2018/04/03/attention.html#encoder-and-decoder-stacks
    """
    def __init__(self, size):
        super(SublayerConnection, self).__init__()
        self.norm = torch.nn.LayerNorm(size)

    def forward(self, layer_input, layer):
        return layer_input + layer(self.norm(layer_input))

class SelfAttention(torch.nn.Module):
    """ Self-attn module for Transformer. """
    def __init__(self, dm, dq, dk, dv):
        super(SelfAttention, self).__init__()
        self.dm = dm
        self.dq = dq
        self.dk = dk
        self.dv = dv

        self.q_embedding = torch.nn.Linear(dm, dq)
        self.k_embedding = torch.nn.Linear(dm, dk)
        self.v_embedding = torch.nn.Linear(dm, dm)

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_seq):
        Q, K, V = self.q_embedding(input_seq), self.k_embedding(input_seq), self.v_embedding(input_seq)
        scores = Q.bmm(K.transpose(1, 2))
        scores = self.softmax(scores / np.sqrt(self.dk))
        scores = scores.bmm(V)
        return scores


def Attention(Q, K, V):
    scores = Q.bmm(K.transpose(1, 2))
    scores = torch.nn.functional.softmax(scores / np.sqrt(K.shape[-1]))
    scores = scores.bmm(V)
    return scores


class PositionwiseFeedForward(torch.nn.Module):
    """ Self-attn module for Transformer. """
    def __init__(self):
        super(PositionwiseFeedForward, self).__init__()

class PositionalEncoding(torch.nn.Module):
    """ Self-attn module for Transformer. """
    def __init__(self):
        super(PositionalEncoding, self).__init__()


if __name__ == "__main__":
    dm = 128
    seq = torch.zeros(8, 31, dm)
    attn = SelfAttention(dm, 12, 12, 12)
    out = attn(seq)
    assert out.shape[-1] == 128