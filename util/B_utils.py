import numpy as np
import torch

def BLoss(criterion, feat, targetP):
    return (1.0 - criterion(feat, targetP)).pow(2).sum()

def pairwise_dist(x,eps= 1e-12):
    x_square = x.pow(2).sum(dim=1)
    prod = x @ x.t()
    pdist = (x_square.unsqueeze(1) + x_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
    pdist[range(len(x)), range(len(x))] = 0.
    return pdist

def pairwise_prob(pdist):
    return torch.exp(-pdist)

def hcr_loss(h, g,eps= 1e-12):
    q1, q2 = pairwise_prob(pairwise_dist(h,eps)), pairwise_prob(pairwise_dist(g,eps))
    return -1 * (q1 * torch.log(q2 + eps)).mean() + -1 * ((1 - q1) * torch.log((1 - q2) + eps)).mean()
