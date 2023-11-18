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
from torch.utils.data.distributed import DistributedSampler

import retnet

def set_random_seeds(seed=0):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
class WikiDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, CHUNK_SIZE, name='wikitext-2-raw-v1', split='train'):
        self.data = load_dataset('wikitext', name, split=split)
        print("Length of dataset: ", len(self.data['text']))
        self.data = tokenizer("\n\n".join(self.data['text']), return_tensors='pt').input_ids[0]
        print("Shape of tokens dataset: ", self.data.shape)
        self.CHUNK_SIZE = CHUNK_SIZE
    def __getitem__(self, idx):
        chunk= self.data[idx * self.CHUNK_SIZE:(idx + 1) * self.CHUNK_SIZE]
        return chunk[:-1], chunk[1:]
    def __len__(self):
        return len(self.data) // self.CHUNK_SIZE
def evaluate(model, data_loader, vocab_size, nsamples=40):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    nll = 0.0
    counter = 0
    
    with torch.no_grad():
        for x,target in data_loader:
            if counter == nsamples:
                break
            counter += 1
            x,target=x.to(model.device),target.to(model.device)
            pred = model(x)
            loss = criterion(pred.view(-1, vocab_size), target.view(-1))
            nll += loss.item()
    
    return np.exp( nll / (nsamples))
prompts = [
    'What is the capital of ',
    'Hi, could you tell me about yourself?',
    'What is the meaning of life?',
    '1 + 1 = ',
]

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--dmodel', type=int, default=768)
    parser.add_argument('--dffn', type=int, default=1536)
    parser.add_argument('--nlayer', type=int, default=12)
    parser.add_argument('--nheads', type=int, default=12)
    parser.add_argument('--chunksize', type=int, default=2048)
    parser.add_argument('--randomseed', type=int, default=0)
    parser.add_argument('--chunksize', type=int, default=2048)
    parser.add_argument('--twoorthree', type=int, default=3)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seeds(args.randomseed)
    
    
    
    
    
    layers = args.nlayer
    hidden_dim = args.dmodel
    ffn_size = args.dffn
    heads = args.nheads
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    vocab_size = len(tokenizer)
    drop_prob = args.dropprob
    
    net = retnet.RetNet(layers, hidden_dim, ffn_size, heads, len(tokenizer), drop_prob, double_v_dim=False).to(device)
    net.device = device
    
    state_dict = torch.load(f"{args.ckpt}.pth")
    print(state_dict)
    net.load_state_dict(state_dict)
    
    
    # dataset
    CHUNK_SIZE = args.chunksize
    time_data = time.time()
    datasetname = 'wikitext-2-raw-v1' if args.twoorthree == 2 else 'wikitext-103-raw-v1'
    # train_set = WikiDataset(tokenizer, CHUNK_SIZE, datasetname, 'train')
    # val_set = WikiDataset(tokenizer, CHUNK_SIZE, datasetname, 'validation')
    test_set = WikiDataset(tokenizer, CHUNK_SIZE, datasetname, 'test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, pin_memory=True, num_workers=8)
    print(f"Time to load data (device {device}): {time.time() - time_data}")
    
    
    start_time = time.time()
    test_ppl = evaluate(net, test_loader, vocab_size, nsamples=len(test_loader))
    print(f"Time to evaluate {len(test_loader)} samples (device {device}): {time.time() - start_time}")

    print(f"Test perplexity: {test_ppl}")
    
    # generation too hard
    # for prompt in prompts:
    #     print(prompt)
    #     input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    #     max_length = input_ids.shape[1] + 100
    #     output = net.generate(input_ids=input_ids, max_length=max_length)
    
    