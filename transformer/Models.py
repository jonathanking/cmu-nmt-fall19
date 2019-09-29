import torch
from .Encoder import Encoder
from .Decoder import Decoder

class Transformer(torch.nn.Module):
    """ Transformer based model. """

    def __init__(self, din, dm, dq, dk, dv, dff, dout, n_enc_layers, n_dec_layers, vocab_size):
        super(Transformer, self).__init__()
        self.din = din
        self.dm = dm
        self.dq = dq
        self.dk = dk
        self.dv = dv
        self.dff = dff
        self.dout = dout
        self.n_enc_layers = n_enc_layers
        self.n_dec_laers = n_dec_layers
        self.vocab_size = vocab_size

        self.encoder = Encoder(din, dm, dq, dk, dv, dff, n_enc_layers, self.vocab_size)
        self.decoder = Decoder(dm, dq, dk, dv, dff, n_dec_layers, self.vocab_size)
        self.output_projection = torch.nn.Linear(dm, dout)
        self.output_softmax = torch.nn.Softmax()

    def forward(self, enc_input, dec_input):
        enc_output, enc_attn = Encoder(enc_input)
        dec_output = Decoder(dec_input, enc_output, enc_attn)
        logits = self.output_projection(dec_output)
        return self.output_softmax(logits)


