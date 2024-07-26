from __future__ import absolute_import
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.functional import linear, normalize
from scipy.optimize import linear_sum_assignment

class semantics(nn.Module):
    def __init__(self, feat_in, num_classes, gpu_id=0):
        super(semantics, self).__init__()
        self.gpu_id = gpu_id
        self.register_buffer("queue", torch.zeros((num_classes, feat_in)))

    def forward(self, x, labels_a, lam=1):
        with torch.no_grad():
            x = F.normalize(x, p=2, dim=1)

            self.queue[labels_a, :] += 0.9 * self.queue[labels_a, :] + (1 - 0.9) * lam * x
            self.queue = self.queue / torch.clamp(torch.sqrt(torch.sum(self.queue ** 2, dim=1, keepdims=True)), 1e-8)

        return self.queue



class fixed_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, centroid_path, gpu_id=0, fix_bn=False):
        super(fixed_Classifier, self).__init__()
        self.gpu_id = gpu_id
        self.offline_haungarian_idx = None
        self.P = self.get_precomputed_centers(centroid_path).cuda(self.gpu_id)
        self.register_buffer("queue", torch.randn(num_classes, feat_in))
        M, _ = self.haungarian()
        self.polars = M.cuda(gpu_id)

        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)

        self.BN_H = nn.BatchNorm1d(feat_in)
        if fix_bn:
            self.BN_H.weight.requires_grad = False
            self.BN_H.bias.requires_grad = False

    def get_precomputed_centers(self,centroid_path):
        # centers = torch.load('100centers_64dim.pth')['w'].T
        centers = torch.load(centroid_path)['w'].T
        centers = centers / torch.clamp(torch.sqrt(torch.sum(centers ** 2, dim=1, keepdims=True)), 1e-8)

        return centers.T

    def forward(self, x):
        x = self.BN_H(x)
        x = x / torch.clamp(torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        return x

    def forward_momentum(self, x, labels_a, lam=1):
        with torch.no_grad():
            x = F.normalize(x, p=2, dim=1)
            self.queue[labels_a, :] += 0.9 * self.queue[labels_a, :] + (1 - 0.9) * lam * x
            self.queue = self.queue / torch.clamp(torch.sqrt(torch.sum(self.queue ** 2, dim=1, keepdims=True)), 1e-8)

    def haungarian(self):
        score = torch.matmul(self.queue.cuda(self.gpu_id), self.P)
        idx = linear_sum_assignment(-score.cpu().detach().numpy())[1]
        self.offline_haungarian_idx = idx
        kernel = self.P.clone()

        for i, j in enumerate(self.offline_haungarian_idx):
            kernel[:, i] = self.P[:, j]

        return kernel, self.offline_haungarian_idx

    def update_fixed_center(self):
        self.polars, idx = self.haungarian()
        return idx

    def predict(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = torch.mm(x, self.polars)
        return x

    def predictLT(self, x,weighted_polars):
        x = F.normalize(x, p=2, dim=1)
        x = torch.mm(x, weighted_polars)
        return x