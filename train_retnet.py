import tqdm
import argparse
import time
import numpy as np
import math

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from datasets import load_dataset

import retnet

#args
parser = argparse.ArgumentParser(description='Retnet training')
parser.add_argument('--dmodel', type=int, default=768)
parser.add_argument('--dffn', type=int, default=1536)
parser.add_argument('--nlayer', type=int, default=12)
parser.add_argument('--nheads', type=int, default=12)
parser.add_argument('--chunksize', type=int, default=2048)
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--lr', type=int, default=0.001)
parser.add_argument('--numepochs', type=int, default=20)
parser.add_argument('--printevery', type=int, default=100)
args = parser.parse_args()

#model
layers = args.nlayer
hidden_dim = args.dmodel
ffn_size = args.dffn
heads = args.nheads
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
vocab_size = len(tokenizer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = retnet.RetNet(layers, hidden_dim, ffn_size, heads, len(tokenizer), double_v_dim=True).to(device)
net.device = device
print(f'Number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}')

# dataset
CHUNK_SIZE = args.chunksize
class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, CHUNK_SIZE=CHUNK_SIZE, name='wikitext-2-raw-v1', split='train'):
        self.data = load_dataset('wikitext', name, split=split)
        self.data = tokenizer("\n\n".join(self.data['text']), return_tensors='pt').input_ids[0]
        self.CHUNK_SIZE = CHUNK_SIZE
    def __getitem__(self, idx):
        chunk= self.data[idx * self.CHUNK_SIZE:(idx + 1) * self.CHUNK_SIZE]
        return chunk[:-1], chunk[1:]
    def __len__(self):
        return len(self.data) // self.CHUNK_SIZE
train_set = WikiDataset(tokenizer, CHUNK_SIZE, 'wikitext-2-raw-v1', 'train')
val_set = WikiDataset(tokenizer, CHUNK_SIZE, 'wikitext-2-raw-v1', 'validation')

# evaluation
def evaluate(model, nsamples=40):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    nll = 0.0
    counter = 0
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)
    for x,target in val_loader:
        if counter == nsamples:
            break
        counter += 1
        x,target=x.to(model.device),target.to(model.device)
        with torch.no_grad():
            pred = model(x)
        loss = criterion(pred.view(-1, vocab_size), target.view(-1))
        nll += loss.item()
    
    model.train()
    return math.exp( nll / (nsamples))

# training
BATCH_SIZE = args.batchsize
LR = args.lr
EPOCHS = args.numepochs
PRINT_EVERY = args.printevery
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(reduction='mean')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

start = time.time()
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for x,target in tqdm.tqdm(train_loader):
        x,target = x.to(device),target.to(device)

        optimizer.zero_grad()
        pred = net(x)
        loss = criterion(pred.view(-1, vocab_size), target.view(-1))
        # assert loss.ndim == 0
        loss.backward()
        optimizer.step()
        if (epoch % PRINT_EVERY) == 0:
            print(f"Loss: {loss.item()}")
    
    print(f"Validation perplexity: {evaluate(net):.3f}")

torch.save(net.state_dict(), 'retnet_{layers}_{hidden_dim}_{ffn_size}_{heads}.pth')
torch.cuda.synchronize()
print(f'Transformer per-epoch training time: {(time.time() - start) / EPOCHS}')
# test evaluation
net.eval()
test_set = WikiDataset(tokenizer, CHUNK_SIZE, 'wikitext-2-raw-v1', 'test')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
total_loss = 0
with torch.no_grad():
    for x,target in tqdm.tqdm(test_loader):
        x,target = x.to(device),target.to(device)
        pred = net(x)
        loss = criterion(pred.view(-1, vocab_size), target.view(-1))
        total_loss += loss.item()
print(f"Test perplexity {np.exp(total_loss / len(test_loader)  )}")