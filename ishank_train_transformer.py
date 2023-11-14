import torch
from torch.utils import data
from torch import nn, optim
from torch.nn import functional as F

from transformers import GPT2Tokenizer
from datasets import load_dataset

import numpy as np

from tqdm import tqdm

from transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
Tweak these
"""

NUM_LAYERS = 4
NUM_HEADS = 8
HIDDEN_DIM = 768
FFN_SIZE = 4 * HIDDEN_DIM

MAX_SEQ_LENGTH = 128


DROPOUT = 0.2
LEARNING_RATE = 1e-5
BATCH_SIZE = 32

class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, CHUNK_SIZE=MAX_SEQ_LENGTH, name='wikitext-2-raw-v1', split='train'):
        self.data = load_dataset('wikitext', name, split=split)
        self.data = tokenizer("\n\n".join(self.data['text']), return_tensors='pt').input_ids[0]
        self.CHUNK_SIZE = CHUNK_SIZE
    def __getitem__(self, idx):
        chunk= self.data[idx * self.CHUNK_SIZE:(idx + 1) * self.CHUNK_SIZE]
        return chunk[:-1], chunk[1:]
    def __len__(self):
        return len(self.data) // self.CHUNK_SIZE

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')



train_set = WikiDataset(tokenizer, MAX_SEQ_LENGTH, 'wikitext-2-raw-v1', 'train')
val_set = WikiDataset(tokenizer, MAX_SEQ_LENGTH, 'wikitext-2-raw-v1', 'validation')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)


transformer = Transformer(NUM_LAYERS, HIDDEN_DIM, FFN_SIZE, NUM_HEADS, len(tokenizer), DROPOUT).to(device)
optimizer = optim.Adam(transformer.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(reduction='mean')


if __name__ == '__main__':
    count = 0
    n_epochs = 10
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        for input, target in tqdm(train_loader):
            input, target = input.to(device), target.to(device)
            mask = torch.ones(input.shape[0], MAX_SEQ_LENGTH, MAX_SEQ_LENGTH).tril().to(device)
            optimizer.zero_grad()
            pred = transformer(input)
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
            pred = transformer(input)
            loss = criterion(pred.transpose(1, 2), target)
            total_loss += loss.item()
            count += 1

        print(f"Validation perplexity {np.exp((total_loss / count))}")