import torch
import torch.nn as nn
import numpy as np
import retnet
from transformers import GPT2Tokenizer
from datasets import load_dataset
import tqdm

#model
layers = 2
hidden_dim = 300
ffn_size = hidden_dim * 2
heads = 3
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
vocab_size = len(tokenizer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = retnet.RetNet(layers, hidden_dim, ffn_size, heads, len(tokenizer), double_v_dim=True).to(device)
print(f'Number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}')

# evaluate
def evaluate(model, tokenizer, nsamples=40):
    testenc = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    testenc = tokenizer("\n\n".join(testenc['text']), return_tensors='pt')

    testenc = testenc.input_ids.to(model.device)
    model = model.eval()

    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * 2048):((i + 1) * 2048)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * 2048):((i + 1) * 2048)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))

# dataset
CHUNK_SIZE = 2048
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

# training
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 50
PRINT_EVERY = 100
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(reduction='mean')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)


for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for x,target in tqdm.tqdm(train_loader):
        x,target = x.to(device),target.to(device)

        optimizer.zero_grad()
        pred = net(x)
        loss = criterion(pred.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()
        if (epoch % PRINT_EVERY) == 0:
            print(f"Loss: {loss.item()}")
    
    print(f"Validation perplexity: {evaluate(net, tokenizer)}")

torch.save(net.state_dict(), 'retnet.pth')

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