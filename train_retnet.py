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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
def set_random_seeds(seed=0):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    
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

def main(args):
    try:
        set_random_seeds(args.randomseed)
        if args.isdistributed==1:
            dist.init_process_group("nccl")
            rank = dist.get_rank()
            device = rank % torch.cuda.device_count()
        else:
            rank = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #model
        layers = args.nlayer
        hidden_dim = args.dmodel
        ffn_size = args.dffn
        heads = args.nheads
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        vocab_size = len(tokenizer)
        drop_prob = args.dropprob
        
        time_model_loading_start = time.time()
        net = retnet.RetNet(layers, hidden_dim, ffn_size, heads, len(tokenizer), drop_prob, double_v_dim=False).to(device)
        net.device = device
        if args.isdistributed==1:
            net = DDP(net, device_ids=[device])
        if rank == 0:
            print(f'Number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}')
        if args.resumefilename != '' and args.isdistributed==1:
            map_location = {'cuda:0': f'cuda:{device}'}
            state_dict = torch.load(args.resumefilename, map_location=map_location)
            net.load_state_dict(state_dict)
        print(f"Time to load model (device {device}): {time.time() - time_model_loading_start}")

        # dataset
        CHUNK_SIZE = args.chunksize
        time_data = time.time()
        datasetname = 'wikitext-2-raw-v1' if args.twoorthree == 2 else 'wikitext-103-raw-v1'
        train_set = WikiDataset(tokenizer, CHUNK_SIZE, datasetname, 'train')
        val_set = WikiDataset(tokenizer, CHUNK_SIZE, datasetname, 'validation')
        test_set = WikiDataset(tokenizer, CHUNK_SIZE, datasetname, 'test')
        print(f"Time to load data (device {device}): {time.time() - time_data}")

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
        
        best_val_ppl = float('inf')

        optimizer = torch.optim.AdamW(net.parameters(), lr=LR1, betas=(BETA1, BETA2), weight_decay=WEIGHTDECAY)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        
        time_train_loader = time.time()
        train_sampler = DistributedSampler(train_set)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=True, num_workers=8)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, pin_memory=True, num_workers=8)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, pin_memory=True, num_workers=8)
        
        print(f"Time to create train, val, test loader and sampler (device {device}): {time.time() - time_train_loader}")

        print(f"Length of train set over batch size (device {device}): {len(train_set) // BATCH_SIZE}")
        print(f"Length of train_loader (device {device}): {len(train_loader)}")
            

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            time_epoch_start = time.time()
            count = 0
            if rank == 0:
                validation_start_time = time.time()
                val_ppl = evaluate(net, val_loader, vocab_size, nsamples=len(val_set))
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    torch.save(net.state_dict(), f'{args.savenamebest}.pth')
                print(f"Time to validate: {time.time() - validation_start_time}")
                print(f"Validation perplexity: {val_ppl:.3f}")
            if args.isdistributed==1:
                dist.barrier()
            print(f"Starting training on device {device}")
            net.train()
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
                    print(f"Loss (device {device}): {loss.item()}")
                count += 1
            
            if rank == 0:
                time_save_final_start = time.time()
                torch.save(net.state_dict(), f'{args.savenamefinal}.pth')
                print("Time to save final model: ", time.time() - time_save_final_start)
      
            
            print(f"Time to train epoch (device {device}): {time.time() - time_epoch_start}")

        
        if rank == 0:
            time_test_start = time.time()
            test_ppl = evaluate(net, test_loader, vocab_size, nsamples= len(test_set))
            print(f"Test perplexity {test_ppl }")
            print(f"Time to test: {time.time() - time_test_start}")
            
    except Exception as e:
        print(f"Error in training: {e}")
    finally:
        if args.isdistributed == 1:
            dist.barrier()
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
    parser.add_argument('--savenamebest', type=str, default='retnet_best')
    parser.add_argument('--savenamefinal', type=str, default='retnet_final')
    parser.add_argument('--resumefilename', type=str, default='')
    parser.add_argument('--randomseed', type=int, default=0)
    args = parser.parse_args()
    main(args)