import torch
import numpy as np


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
    def __init__(self, dm, dh):
        super(PositionwiseFeedForward, self).__init__()
        self.dm = dm
        self.dh = dh
        self.layer1 = torch.nn.Linear(dm, dh)
        self.layer2 = torch.nn.Linear(dh, dm)
        self.relu = torch.nn.ReLU()
        # TODO: is this implementation with linear layers accurate?
        # self.conv1d_1 = torch.nn.conv1d(dm, dh)
        # self.conv1d_2 = torch.nn.conv1d(dh, dm)

    def forward(self, input_seq):
        return self.layer2(self.relu(self.layer1(input_seq)))


class PositionalEncoding(torch.nn.Module):
    """ Positional encoding layer for the Transformer model.
    # TODO I think jalammar's position encoding viz is wrong. It doesn't alternate sin/cos.
    """
    def __init__(self, dm, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dm = dm
        self.max_seq_len = max_seq_len
        self.pos_encodings = self.create_encodings()

    def forward(self, seq):
        return self.pos_encodings[0:seq.shape[1]]

    def create_encodings(self):
        return torch.stack([self.get_pos_enc_for_word(i) for i in range(self.max_seq_len)])

    def get_pos_enc_for_word(self, pos):
        enc = torch.zeros(self.dm)
        for i in range(self.dm // 2):
            denominator = 10000**(2*i/self.dm)
            enc[2*i] = np.sin(pos/denominator)
            enc[2*i + 1] = np.cos(pos/denominator)
        return enc


if __name__ == "__main__":
    seq = torch.ones(8, 7, 64)
    penc = PositionalEncoding(64, 300)
    print(penc(seq).shape)
    print(penc(seq))
