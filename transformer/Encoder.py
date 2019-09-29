import torch
from .Layers import SelfAttention, PositionwiseFeedForward, PositionalEncoding

class Encoder(torch.nn.Module):
    """ Transformer encoder model. """

    def __init__(self, din, dm, dq, dk, dv, dff, n_enc_layers, vocab_size):
        super(Encoder, self).__init__()
        self.din = din
        self.dm = dm
        self.dq = dq
        self.dk = dk
        self.dv = dv
        self.dff = dff
        self.vocab_size = vocab_size
        self.n_enc_layers = n_enc_layers

        self.input_embedding = torch.nn.Embedding(self.vocab_size, self.dm)
        self.positional_enc = PositionalEncoding()
        self.self_attn = SelfAttention()
        self.pwff = PositionwiseFeedForward()

        self.enc_layers = [EncoderLayer(dm, dq, dk, dv, dff) for _ in self.n_enc_layers]

    def forward(self, src_seq):
        enc_output = self.input_embedding(src_seq)
        enc_output = enc_output + self.positional_enc(src_seq)
        for enc_layer in self.enc_layers:
            enc_output, enc_attn = enc_layer(enc_output)
        return enc_output, enc_attn


class EncoderLayer(torch.nn.Module):
    """ Transformer encoder layer. """

    def __init__(self, dm, dq, dk, dv, dff):
        super(EncoderLayer, self).__init__()
        self.dm = dm
        self.dq = dq
        self.dk = dk
        self.dv = dv
        self.dff = dff

        self.self_attn = SelfAttention()
        self.pwff = PositionwiseFeedForward()

    def forward(self, src_seq):
        enc_output, enc_attn = self.self_attn(src_seq)
        enc_output = self.pwff(enc_output)
        return enc_output, enc_attn