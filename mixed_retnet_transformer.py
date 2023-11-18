import torch
import torch.nn as nn

from retention import MultiScaleRetention
from transformer import MultiHeadedAttention

class MixedRetNetTransformer(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, vocab_size, dropout, binary_vector, double_v_dim=False):
        super(MixedRetNetTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.binary_vector = binary_vector
        self.register_buffer("mask", None)
        
        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim) if binary_vector[i]=="1" else MultiHeadedAttention(hidden_dim, heads)
            for i in range(layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
    
    def forward(self, X: torch.Tensor):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        X = self.embed(X)
        if self.mask is None or self.mask.shape[0] != X.shape[1]:
            self.mask = torch.ones(X.shape[1], X.shape[1]).tril().to(X.device)
        for i in range(self.layers):
            if self.binary_vector[i] == "1":
                Y = self.retentions[i](self.layer_norms_1[i](X))
            else:
                Y = self.retentions[i](self.layer_norms_1[i](X), self.mask)
            Y = self.dropout(Y)
            Y = Y + X
            Z = self.ffns[i](self.layer_norms_2[i](Y))
            Z = self.dropout(Z)
            X = Z + Y
        X = self.dropout(X)
        X = self.proj(X)
        return X

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        X: (batch_size, sequence_length, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        s_ns = []
        for i in range(self.layers):
            # list index out of range
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n), s_n_1s[i], n)
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n
        
        return x_n, s_ns
    
    def forward_chunkwise(self, x_i, r_i_1s, i):
        """`
        X: (batch_size, sequence_length, hidden_size)
        r_i_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        r_is = []
        for j in range(self.layers):
            o_i, r_i = self.retentions[j].forward_chunkwise(self.layer_norms_1[j](x_i), r_i_1s[j], i)
            y_i = o_i + x_i
            r_is.append(r_i)
            x_i = self.ffns[j](self.layer_norms_2[j](y_i)) + y_i
        
        return x_i, r_is
    
    # generation is a little scuffed since we don't train with start/end tokens
    @torch.inference_mode
    def generate(self, X: torch.Tensor, max_tokens=100):
        # X is shape batch, sequence length
        initial_seq_len = X.shape[1]
        
        for i in range(initial_seq_len):
            pass
         
        