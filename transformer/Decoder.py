import torch
from .Layers import SelfAttention, PositionwiseFeedForward, PositionalEncoding

class Decoder(torch.nn.Module):
    """ Transformer decoder model. """

    def __init__(self, dout, dm, dq, dk, dv, dff, n_dec_layers):
        super(Decoder, self).__init__()
        self.dm = dm
        self.dq = dq
        self.dk = dk
        self.dv = dv
        self.dff = dff
        self.dout = dout
        self.n_dec_layers = n_dec_layers

        self.positional_enc = PositionalEncoding()
        self.input_embedding = torch.nn.Embedding(self.dout, self.dm)
        self.dec_layers = [DecoderLayer(dm, dq, dk, dv, dff) for _ in self.n_dec_layers]

    def forward(self, dec_input, enc_output):
        dec_output = self.input_embedding(dec_input)
        dec_output = dec_output + self.positional_enc(dec_output)
        for dec_layer in self.dec_layers:
            dec_output = dec_layer(dec_output, enc_output)
        return dec_output


class DecoderLayer(torch.nn.Module):
    """ Transformer decoder layer. """

    def __init__(self, dm, dq, dk, dv, dff):
        super(DecoderLayer, self).__init__()
        self.dm = dm
        self.dq = dq
        self.dk = dk
        self.dv = dv
        self.dff = dff

        self.self_attn = SelfAttention() # TODO use mask
        self.pwff = PositionwiseFeedForward()
        self.layer_norm = torch.nn.LayerNorm()

    def forward(self, dec_layer_input, enc_output):
        dec_output = dec_layer_input + self.self_attn(self.layer_norm(dec_layer_input))        # Self-attn
        dec_output = dec_output + self.layer_norm(self.enc_dec_attn(dec_output, enc_output))   # Enc/dec-attn
        dec_output =  dec_output + self.layer_norm(self.pwff(dec_output))                      # Feed forward
        return dec_output