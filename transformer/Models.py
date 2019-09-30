import torch
from .Encoder import Encoder
from .Decoder import Decoder

class Transformer(torch.nn.Module):
    """ Transformer based model. """

    def __init__(self, dm, dff, vocab_size, n_heads, n_enc_layers, n_dec_layers):
        super(Transformer, self).__init__()
        self.din = vocab_size
        self.dout = vocab_size
        self.dm = dm
        self.dff = dff
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.n_dec_laers = n_dec_layers


        self.encoder = Encoder(self.din, dm, dff, n_heads, n_enc_layers)
        self.decoder = Decoder(self.dout, dm, dff, n_heads, n_dec_layers)
        self.output_projection = torch.nn.Linear(dm, self.dout)
        self.output_softmax = torch.nn.Softmax()

    def forward(self, enc_input, dec_input):
        enc_output, enc_attn = self.encoder(enc_input)
        dec_output = self.decoder(dec_input, enc_output, enc_attn)
        logits = self.output_projection(dec_output)
        return self.output_softmax(logits)


