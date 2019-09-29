import torch


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

if __name__ == "__main__":
    dm = 128
    seq = torch.zeros(8, 31, dm)
    attn = SelfAttention(dm, 12, 12, 12)
    out = attn(seq)
    assert out.shape[-1] == 128