import tqdm
import argparse
import time
import numpy as np
import os

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from datasets import load_dataset

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


import retnet

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def warmup_scheduler(optimizer, warmup_steps, initial_lr, target_lr, step):
    if step < warmup_steps:
        lr = initial_lr + (target_lr - initial_lr) * step / warmup_steps
    else:
        lr = target_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, CHUNK_SIZE, name='wikitext-2-raw-v1', split='train'):
        self.data = load_dataset('wikitext', name, split=split)
        self.data = tokenizer("\n\n".join(self.data['text']), return_tensors='pt').input_ids[0]
        self.CHUNK_SIZE = CHUNK_SIZE
    def __getitem__(self, idx):
        chunk= self.data[idx * self.CHUNK_SIZE:(idx + 1) * self.CHUNK_SIZE]
        return chunk[:-1], chunk[1:]
    def __len__(self):
        return len(self.data) // self.CHUNK_SIZE
def evaluate(model, dataset, vocab_size, nsamples=40):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    nll = 0.0
    counter = 0
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
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
    return np.exp( nll / (nsamples))

def main(args, rank, world_size):
    if args.isdistributed==1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()

    #model
    layers = args.nlayer
    hidden_dim = args.dmodel
    ffn_size = args.dffn
    heads = args.nheads
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    vocab_size = len(tokenizer)
    drop_prob = args.dropprob
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    net = retnet.RetNet(layers, hidden_dim, ffn_size, heads, len(tokenizer), drop_prob, double_v_dim=True).to(device)
    net.device = device
    print(f'Number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}')

    # dataset
    CHUNK_SIZE = args.chunksize
    
    datasetname = 'wikitext-2-raw-v1' if args.twoorthree == 2 else 'wikitext-103-raw-v1'
    train_set = WikiDataset(tokenizer, CHUNK_SIZE, datasetname, 'train')
    val_set = WikiDataset(tokenizer, CHUNK_SIZE, datasetname, 'validation')

    # training
    BATCH_SIZE = args.batchsize
    LR1 = args.lr1
    LR2 = args.lr2
    EPOCHS = args.numepochs
    PRINT_EVERY = args.printevery
    # optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    BETA1 = args.beta1
    BETA2 = args.beta2
    WEIGHTDECAY = args.weightdecay
    WARMUP_STEPS = args.warmupsteps
    
    best_val_ppl = 1e9
    best_model = None

    optimizer = torch.optim.AdamW(net.parameters(), lr=LR1, betas=(BETA1, BETA2), weight_decay=WEIGHTDECAY)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    start = time.time()
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        count = 0
        for x,target in tqdm.tqdm(train_loader):
            x,target = x.to(device),target.to(device)
            
            global_step = epoch * len(train_loader) + count
            warmup_scheduler(optimizer, WARMUP_STEPS, LR1, LR2, global_step)

            optimizer.zero_grad()
            pred = net(x)
            loss = criterion(pred.view(-1, vocab_size), target.view(-1))
            # assert loss.ndim == 0
            loss.backward()
            optimizer.step()
            if (count % PRINT_EVERY) == 0:
                print(f"Loss: {loss.item()}")
            count += 1
            if count == 10:
                break
        val_ppl = evaluate(net, val_set, vocab_size)
        if best_model is None or val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_model = net.cpu()
        print(f"Validation perplexity: {val_ppl:.3f}")

    torch.cuda.synchronize()
    print(f'Transformer per-epoch training time: {(time.time() - start) / EPOCHS}')
    # test evaluation
    test_set = WikiDataset(tokenizer, CHUNK_SIZE, datasetname, 'test')
    test_ppl = evaluate(net, test_set, vocab_size, nsamples= len(test_set))
    best_model.device = device
    test_ppl_best = evaluate(best_model.to(device), test_set, vocab_size, nsamples= len(test_set))
    
    print(f"Test perplexity {test_ppl }")
    print(f"Best test perplexity {test_ppl }")
    torch.save(net.state_dict(), f'retnet_{layers}_{hidden_dim}_{ffn_size}_{heads}_{test_ppl:.3f}_final.pth')
    torch.save(net.state_dict(), f'retnet_{layers}_{hidden_dim}_{ffn_size}_{heads}_{test_ppl_best:.3f}_best.pth')
    if args.isdistributed == 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    #args
    parser = argparse.ArgumentParser(description='Retnet training')
    parser.add_argument('--dmodel', type=int, default=768)
    parser.add_argument('--dffn', type=int, default=1536)
    parser.add_argument('--nlayer', type=int, default=12)
    parser.add_argument('--nheads', type=int, default=12)
    parser.add_argument('--chunksize', type=int, default=2048)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--lr1', type=float, default=0.01)
    parser.add_argument('--lr2', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--weightdecay', type=float, default=0.05)
    parser.add_argument('--warmupsteps', type=int, default=375)
    parser.add_argument('--dropprob', type=float, default=0.1)
    parser.add_argument('--numepochs', type=int, default=20)
    parser.add_argument('--printevery', type=int, default=100)
    parser.add_argument('--twoorthree', type=int, default=3)
    parser.add_argument('--isdistributed', type=int, default=0)
    args = parser.parse_args()
    main(args, 0, 0)