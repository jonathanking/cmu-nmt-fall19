import torch
from .Encoder import Encoder
from .Decoder import Decoder
from .Attention import subsequent_mask


class Transformer(torch.nn.Module):
    """ Transformer based model. """
    # TODO implement dropout for all layers

    def __init__(self, dm, dff, vocab_size, n_heads, n_enc_layers, n_dec_layers, max_seq_len, pad_char):
        super(Transformer, self).__init__()
        self.din = vocab_size
        self.dout = vocab_size
        self.dm = dm
        self.dff = dff
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.n_dec_laers = n_dec_layers
        self.max_seq_len = max_seq_len
        self.pad_char = pad_char


        self.encoder = Encoder(self.din, dm, dff, n_heads, n_enc_layers, max_seq_len)
        self.decoder = Decoder(self.dout, dm, dff, n_heads, n_dec_layers, max_seq_len)
        self.output_projection = torch.nn.Linear(dm, self.dout)
        self.output_softmax = torch.nn.Softmax()

    def forward(self, enc_input, dec_input):
        src_mask = (enc_input != self.pad_char) # TODO is this mask the right dimensionality?
        tgt_mask = subsequent_mask(dec_input.shape[1])
        enc_output = self.encoder(enc_input, src_mask)
        dec_output = self.decoder(dec_input, enc_output, tgt_mask, src_mask)
        logits = self.output_projection(dec_output)
        return self.output_softmax(logits)


