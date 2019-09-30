import torch
from .Sublayers import PositionwiseFeedForward, PositionalEncoding, SublayerConnection
from .Attention import MultiHeadedAttention

class Encoder(torch.nn.Module):
    """ Transformer encoder model. """

    def __init__(self, din, dm, dff, n_heads, n_enc_layers, max_seq_len):
        super(Encoder, self).__init__()
        self.din = din
        self.dm = dm
        self.dff = dff
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.max_seq_len = max_seq_len

        self.input_embedding = torch.nn.Embedding(self.din, self.dm)
        self.positional_enc = PositionalEncoding(dm, max_seq_len)

        self.enc_layers = [EncoderLayer(dm, dff, n_heads) for _ in self.n_enc_layers]

    def forward(self, src_seq):
        enc_output = self.input_embedding(src_seq)
        enc_output = enc_output + self.positional_enc(enc_output)
        for enc_layer in self.enc_layers:
            enc_output = enc_layer(enc_output)
        return enc_output


class EncoderLayer(torch.nn.Module):
    """ Transformer encoder layer. """

    def __init__(self, dm, dff, n_heads):
        super(EncoderLayer, self).__init__()
        self.dm = dm
        self.dff = dff
        self.n_heads = n_heads

        self.self_attn = MultiHeadedAttention(dm, n_heads)
        self.pwff = PositionwiseFeedForward(dm, dff)
        self.sublayer_connections = [SublayerConnection(dm) for _ in range(2)]

    def forward(self, enc_layer_input):
        enc_output = self.sublayer_connections[0](enc_layer_input, self.self_attn)
        enc_output = self.sublayer_connections[1](enc_output, self.pwff)
        return enc_output