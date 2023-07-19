
import os
import sys
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn, einsum 
from einops import rearrange, repeat

import pandas as pd

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., adj_dim = 83):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads # n- # of tokens, h- # of heads, n- # of dimensions for each head
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots) 
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



#ConvEncoder#
class LinearEncoder(nn.Module):
    def __init__(self, x_dim=28*28, hidden_dims=[512, 256], latent_dim=2, in_channels=1):
        super().__init__()
        self.fc1 = nn.Linear(x_dim*in_channels, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc31 = nn.Linear(hidden_dims[1], latent_dim) # mu
        self.fc32 = nn.Linear(hidden_dims[1], latent_dim) # log_var       
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var

class ConvEncoder(nn.Module):
    def __init__(self, x_dim=28*28, hidden_dims=[32, 64], latent_dim=2, in_channels=1, activation=nn.ReLU):
        super().__init__()
        self.activation = activation
        modules = []
        '''modules.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=hidden_dims[0],
                              kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(hidden_dims[0]),
                    activation()))
        modules.append(nn.Sequential(
                    nn.Conv2d(hidden_dims[0], out_channels=2*hidden_dims[0],
                              kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm2d(2*hidden_dims[0]),
                    activation()))
        in_channels = 2*hidden_dims[0]'''
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm2d(h_dim),
                    activation())
            )
            in_channels = h_dim
        #bottleneck_res = [28, 14, 7, 4, 2] + [1]*30 ## TODO only valid for 28^2 mnist digits
        bottleneck_res = [int(np.ceil(np.sqrt(x_dim) * 0.5**i)) for i in range(35)] # set res to decrease geometrically
        self.res_flattened = bottleneck_res[len(hidden_dims)]
        self.encoder = nn.Sequential(*modules)
        self.fc_mu =  nn.Linear(hidden_dims[-1]*(self.res_flattened**2), latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*(self.res_flattened**2), latent_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        try:
            mu = self.fc_mu(x)
        except:
            import ipdb; ipdb.set_trace()
        log_var = self.fc_var(x)

        return mu, log_var
        





