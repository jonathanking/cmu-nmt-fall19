import torch
from .Encoder import Encoder
from .Decoder import Decoder

class Transformer(torch.nn.Module):
    """ Transformer based model. """

    def __init__(self, dm, dq, dk, dv, dff, vocab_size, n_enc_layers, n_dec_layers):
        super(Transformer, self).__init__()
        self.din = vocab_size
        self.dm = dm
        self.dq = dq
        self.dk = dk
        self.dv = dv
        self.dff = dff
        self.dout = vocab_size
        self.n_enc_layers = n_enc_layers
        self.n_dec_laers = n_dec_layers

        self.encoder = Encoder(self.din, dm, dq, dk, dv, dff, n_enc_layers)
        self.decoder = Decoder(self.dout, dm, dq, dk, dv, dff, n_dec_layers)
        self.output_projection = torch.nn.Linear(dm, self.dout)
        self.output_softmax = torch.nn.Softmax()

    def forward(self, enc_input, dec_input):
        enc_output, enc_attn = Encoder(enc_input)
        dec_output = Decoder(dec_input, enc_output, enc_attn)
        logits = self.output_projection(dec_output)
        return self.output_softmax(logits)


