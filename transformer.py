import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
NEG_INF = -1e9

class AttentionHead(nn.Module):
    def __init__(self, input_size, hidden_size):
      super().__init__()

      self.input_size = input_size
      self.hidden_size = hidden_size
      self.sqrt_hidden_size = np.sqrt(hidden_size)

      self.linear_Q = nn.Linear(input_size, hidden_size)
      self.linear_K = nn.Linear(input_size, hidden_size)
      self.linear_V = nn.Linear(input_size, hidden_size)
    

    def forward(self, input, mask):
      """
      Input has shape (batch_size, MAX_SEQ_LENGTH, input_size)
      """
      batch_size, seq_length, _ = input.shape

      # TODO
      # use @cache

      queries = self.linear_Q(input)
      keys = self.linear_K(input)
      values = self.linear_V(input)

      scores = (queries @ keys.transpose(1, 2)) / self.sqrt_hidden_size
      scores = scores.masked_fill(mask == 0, NEG_INF)

      probs = F.softmax(scores, dim=-1)
      output = probs @ values

      return output


class MultiHeadedAttention(nn.Module):
  def __init__(self, hidden_size, num_heads):
    super().__init__()

    self.hidden_size = hidden_size
    self.num_heads = num_heads

    self.heads = nn.ModuleList([
        AttentionHead(hidden_size, int(hidden_size / num_heads))
        for _ in range(num_heads)
    ])

    self.proj = nn.Linear(hidden_size, hidden_size)

  def forward(self, input, mask):
    """
    Input has shape (batch_size, MAX_SEQ_LENGTH, input_size)
    """
    return self.proj(torch.concat([head(input, mask) for head in self.heads], dim=-1))

class Transformer(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, vocab_size, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", None)
        self.attentions = nn.ModuleList([
            MultiHeadedAttention(hidden_dim, heads)
            for _ in range(layers)
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

    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        X = self.embed(X)
        if self.mask is None or self.mask.shape[0] != X.shape[1]:
            self.mask = torch.ones(X.shape[1], X.shape[1]).tril().to(X.device)
        for i in range(self.layers):
            Y = self.attentions[i](self.layer_norms_1[i](X), self.mask)
            Y = self.dropout(Y)
            Y = Y + X
            Z = self.ffns[i](self.layer_norms_2[i](Y))
            Z = self.dropout(Z)
            X = Z + Y
        X = self.dropout(X)
        X = self.proj(X)
        return X