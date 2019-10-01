import torch
from .Sublayers import PositionwiseFeedForward, PositionalEncoding, SublayerConnection
from .Attention import MultiHeadedAttention, subsequent_mask

class Decoder(torch.nn.Module):
    """ Transformer decoder model. """

    def __init__(self, dout, dm, dff, n_heads, n_dec_layers, max_seq_len):
        super(Decoder, self).__init__()
        self.dout = dout
        self.dm = dm
        self.dff = dff
        self.n_heads = n_heads
        self.n_dec_layers = n_dec_layers

        self.positional_enc = PositionalEncoding(dm, max_seq_len)
        self.input_embedding = torch.nn.Embedding(self.dout, self.dm)
        self.dec_layers = [DecoderLayer(dm, dff, n_heads) for _ in self.n_dec_layers]

    def forward(self, dec_input, enc_output, tgt_mask, src_mask):
        dec_output = self.input_embedding(dec_input)
        dec_output = dec_output + self.positional_enc(dec_output)
        for dec_layer in self.dec_layers:
            dec_output = dec_layer(dec_output, enc_output, tgt_mask, src_mask)
        return dec_output


class DecoderLayer(torch.nn.Module):
    """ Transformer decoder layer. """

    def __init__(self, dm, dff, n_heads):
        super(DecoderLayer, self).__init__()
        self.dm = dm
        self.dff = dff
        self.n_heads = n_heads

        self.self_attn = MultiHeadedAttention(dm, n_heads)
        self.src_attn = MultiHeadedAttention(dm, n_heads)
        self.pwff = PositionwiseFeedForward(dm, dff)
        self.sublayer_connections = [SublayerConnection(dm) for _ in range(3)]

    def forward(self, dec_input, enc_output, tgt_mask, src_mask):
        dec_output = self.sublayer_connections[0](dec_input, lambda x: self.self_attn(x, x, x, mask=tgt_mask))
        dec_output = self.sublayer_connections[1](dec_output, lambda x: self.src_attn(x, enc_output, enc_output, mask=src_mask))
        dec_output = self.sublayer_connections[2](dec_output, self.pwff)
        return dec_output