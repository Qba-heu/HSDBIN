import os
import random
from pathlib import Path
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.printing import bcolors

class dset(Dataset):
    def __init__(self,n_centroids):
        super(dset, self).__init__()
        self.l = n_centroids
    def __getitem__(self, item):
        return item
    def __len__(self):
        return self.l

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

def uniform_loss(x, t=2):
    return torch.pdist(x.T, p=2).pow(2).mul(-t).exp().mean().log()

class uniform_weights(torch.nn.Module):
    def __init__(self,nclass,embedding_dim):
        super(uniform_weights, self).__init__()
        self.ar_batch = torch.arange(0,nclass)
        self.register_buffer("w",tensor=torch.normal(0, 0.01, (embedding_dim,nclass)))
        self.w.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.w = torch.nn.Parameter(self.w)

    def forward(self,idx,optimizer):
        kernel_norm = l2_norm(self.w[:,idx], axis=0)
        loss = uniform_loss(kernel_norm)
        return loss

    def print_score(self):
        with torch.no_grad():
            kernel_norm = l2_norm(self.w, axis=0)
            scores = torch.mm(kernel_norm.T, kernel_norm)
        return scores.mean().item()

    def print_max_score(self):
        with torch.no_grad():
            kernel_norm = l2_norm(self.w, axis=0)
            scores = torch.mm(kernel_norm.T, kernel_norm)
            ar = torch.arange(scores.shape[0])
            scores[ar, ar] = 0
            max_similarity = torch.max(scores, axis=1)[0]
        return torch.max(max_similarity).item(), max_similarity.mean().item()

    def plt_w(self,title):
        kernel = l2_norm(self.w, axis=0)
        plt.scatter(kernel[0,self.ar_batch].detach().cpu().numpy(),kernel[1,self.ar_batch].detach().cpu().numpy())
        plt.title(title)
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperspherical Prototype Estimation")
    parser.add_argument("--seed", dest="seed", default=0, type=int)
    parser.add_argument("--num_centroids", dest="num_centroids", default=30, type=int)
    parser.add_argument("--batch_size", dest="batch_size", default=100, type=int)
    parser.add_argument("--space_dim", dest="space_dim", default=192, type=int)
    parser.add_argument("--num_epoch", dest="num_epoch", default=100, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # logging
    log_dir = Path("Estimated_prototypes")
    print('=> creating: {}'.format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_writer = (SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard")))

    # initialize prototypes
    start = time.time()
    prototypes = uniform_weights(nclass=args.num_centroids, embedding_dim=args.space_dim).cuda()

    max_cos, mean_max_cos = prototypes.print_max_score()
    print(f'{bcolors.OKBLUE} before training ==> mean cos {prototypes.print_score():.6f}, mean max cos:{mean_max_cos:.6f},  max cos:{max_cos:.6f}{bcolors.ENDC}')

    if args.space_dim == 2:
        max_cos, mean_max_cos = prototypes.print_max_score()
        prototypes.plt_w(f'before training ==> mean cos {prototypes.print_score():.6f}, mean max cos:{mean_max_cos:.6f},  max cos:{max_cos:.6f}')

    # for name, param in prototypes.named_parameters():
    #     print(param.requires_grad, name)
    optim = torch.optim.SGD(params=prototypes.parameters(), lr=0.1)

    data = dset(n_centroids=args.num_centroids)
    dloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    # optimiziation Equations 4, 5
    for i in range(args.num_epoch):

        loss_cumulative = 0
        for batch_idx in dloader:
            optim.zero_grad()
            loss = prototypes(optimizer=optim, idx=batch_idx.cuda())
            loss_cumulative += loss.item()
            loss.backward()
            optim.step()

        max_cos, mean_max_cos = prototypes.print_max_score()
        summary_writer.add_scalar('max cos', max_cos, i)
        summary_writer.add_scalar('mean max cos', mean_max_cos, i)
        summary_writer.add_scalar('loss', loss_cumulative, i)
        summary_writer.add_scalar('time', time.time() - start, i)

        if i % 10 == 0:
            max_cos, mean_max_cos = prototypes.print_max_score()
            print(f'epoch {i} ==> mean cos: {prototypes.print_score():.4f}, mean max cos:{mean_max_cos:.6f},  max cos:{max_cos:.6f}')

    if args.space_dim == 2:
        max_cos, mean_max_cos = prototypes.print_max_score()
        prototypes.plt_w(f'after training ==> mean cos: {prototypes.print_score():.6f},  mean max cos:{mean_max_cos:.6f},  max cos:{max_cos:.6f}')

    max_cos, mean_max_cos = prototypes.print_max_score()
    print(f'{bcolors.OKBLUE}after training ==> mean cos {prototypes.print_score():.6f},  mean max cos:{mean_max_cos:.6f},  max cos:{max_cos:.6f} {bcolors.ENDC}')
    torch.save(prototypes.state_dict(),os.path.join(log_dir,f'{args.num_centroids}centers_{args.space_dim}dim.pth'))