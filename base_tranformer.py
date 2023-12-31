import torch
from torch.utils import data
from torch import nn, optim
from torch.nn import functional as F

import numpy as np

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LENGTH = 128
VOCAB_SIZE = 30000
MIN_OCCURRENCES = 125 # calcualted from vocab_size

TRAIN_PATH = "wikitext-103/wiki.train.tokens"
VAL_PATH = "wikitext-103/wiki.valid.tokens"

SAMPLE_RATIO = 0.1

PAD_INDEX = 0
SOS_INDEX = 1
UNK_INDEX = 2

NEG_INF = -2**31

def get_vocab(path):
  count = dict()
  with open(path) as f:
    for line in f.readlines():
      line = line.strip().split()
      if not line:
        continue
      if line[0] == "=" and line[-1] == "=":
        continue
      for word in line:
        count[word] = count.setdefault(word, 0) + 1
  count["unk"] = 0
  count = {key: value for key, value in count.items() if value >= MIN_OCCURRENCES}
  vocab = sorted(count.items(), key=lambda tup: -tup[1])[:VOCAB_SIZE]
  vocab = [tup[0] for tup in vocab]
  return ["<pad>", "<sos>", "<unk>"] + list(vocab)

def vocab_to_hash(vocab):
  return {el: i for i, el in enumerate(vocab)}

def tokenize(path, vocab_hash):
  corpus = []
  with open(path) as f:
    for line in f.readlines():
      line = line.strip().split()
      if not line:
        continue
      if line[0] == "=" and line[-1] == "=":
        continue
      corpus.extend([SOS_INDEX] + [ vocab_hash[word if word in vocab_hash else "<unk>"] for word in line ])

  return corpus

class LanguageDataset(data.Dataset):
  def __init__(self, train_corpus, vocab, vocab_hash):
    self.corpus = train_corpus[:int(len(train_corpus) * SAMPLE_RATIO)]
    self.vocab = vocab
    self.vocab_hash = vocab_hash

    self.length = int((len(self.corpus) - 1) / MAX_SEQ_LENGTH) - 1

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    if (idx >= self.length):
      raise IndexError
    batch_idx = idx * MAX_SEQ_LENGTH
    sent = self.corpus[batch_idx : batch_idx + MAX_SEQ_LENGTH]
    trg = self.corpus[batch_idx + 1 : batch_idx + MAX_SEQ_LENGTH + 1]
    return torch.tensor(sent), torch.tensor(trg)

class AttentionHead(nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()

    self.input_size = input_size
    self.value_size = hidden_size
    self.key_size = hidden_size
    self.sqrt_key_size = np.sqrt(hidden_size)

    self.linear_Q = nn.Linear(input_size, hidden_size)
    self.linear_K = nn.Linear(input_size, hidden_size)
    self.linear_V = nn.Linear(input_size, hidden_size)


  def forward(self, input, mask):
    """
    Input has shape (batch_size, MAX_SEQ_LENGTH, input_size)
    """
    _, seq_length, _ = input.shape

    queries = self.linear_Q(input)
    keys = self.linear_K(input)
    values = self.linear_V(input)

    scores = (queries @ keys.transpose(1, 2)) / self.sqrt_key_size
    scores = scores.masked_fill(mask == 0, NEG_INF)

    probs = F.softmax(scores, dim=-1)
    output = probs @ values

    return output

class MultiHeadedAttention(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers):
    super().__init__()

    self.heads = nn.ModuleList()
    for _ in range(num_layers):
      self.heads.append(AttentionHead(input_size, hidden_size))

    self.proj = nn.Linear(hidden_size * num_layers, hidden_size)

  def forward(self, input, mask):
    """
    Input has shape (batch_size, MAX_SEQ_LENGTH, input_size)
    """
    return self.proj(torch.concat([head(input, mask) for head in self.heads], dim=-1))

class Transformer(nn.Module):
  def __init__(self, vocab_size, hidden_size, num_layers):
    super().__init__()

    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.embed = nn.Embedding(vocab_size, hidden_size)

    self.attention = MultiHeadedAttention(hidden_size, hidden_size, num_layers)

    self.attention_layer_norm = nn.LayerNorm(hidden_size)

    self.FNN = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size)
      )

    self.FNN_layer_norm = nn.LayerNorm(hidden_size)

    self.proj = nn.Linear(hidden_size, vocab_size)

  def forward(self, ids, mask):
    """
    Input has shape (batch_size, seq_length, input_size)
    """
    input = self.embed(ids)

    output1 = self.attention(input, mask)

    output1 += input
    output1 = self.attention_layer_norm(output1)

    output2 = self.FNN(output1)

    output2 += output1
    output2 = self.FNN_layer_norm(output2)

    return self.proj(output2)

# hyper parameters

batch_size = 64
lr = 0.001
n_epochs = 50

num_transformers = 12
hidden_size = 512

vocab = get_vocab(TRAIN_PATH)
vocab_hash = vocab_to_hash(vocab)

print(len(vocab))

train_tokens = tokenize(TRAIN_PATH, vocab_hash)
val_tokens = tokenize(VAL_PATH, vocab_hash)

train_set = LanguageDataset(train_tokens, vocab, vocab_hash)
train_loader = data.DataLoader(train_set, batch_size=batch_size,
                                      shuffle=True)

val_set = LanguageDataset(val_tokens, vocab, vocab_hash)
val_loader = data.DataLoader(val_set, batch_size=batch_size,
                                      shuffle=True)

transformer = Transformer(len(vocab), hidden_size, 12).to(device)
print(f"Number of parameters: { int(sum([x.numel() for x in transformer.parameters()]) / 1e6) }M")

optimizer = optim.Adam(transformer.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss(reduction='mean')

count = 0
for epoch in range(n_epochs):
  print(f"Epoch {epoch+1}/{n_epochs}")
  for input, target in tqdm(train_loader):
    input, target = input.to(device), target.to(device)
    mask = torch.ones(input.shape[0], MAX_SEQ_LENGTH, MAX_SEQ_LENGTH).tril().to(device)
    optimizer.zero_grad()
    pred = transformer(input, mask)
    loss = criterion(pred.transpose(1, 2), target)
    loss.backward()
    optimizer.step()

  # if (epoch % 5) == 0:
  print("Calculating validation perplexity")
  total_loss = 0
  count = 0
  for input, target in tqdm(val_loader):
    input, target = input.to(device), target.to(device)
    mask = torch.ones(input.shape[0], MAX_SEQ_LENGTH, MAX_SEQ_LENGTH).tril().to(device)
    pred = transformer(input, mask)
    loss = criterion(pred.transpose(1, 2), target)
    total_loss += loss.item()
    count += 1

  print(f"Validation perplexity {np.exp((total_loss / count))}")

  torch.save(transformer.state_dict(), 'transformer.pth')
