#imports
import os
import sys
import math
import random
from multiprocessing import Pool

import numpy as np
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from ete3 import Tree, TextFace, TreeStyle, NodeStyle

def seed_torch(seed=2021):
    """
    Resets the seed for the Python environment, numpy, and torch.
    
    Parameters
    ----------
    seed : int, optional
        default is 2021
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    
    
    
    
    
    
#VAE class
class VariationalAutoencoder(nn.Module):
    """
    A simple VAE model.

    Inherits from nn.Module
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html

    Parameters
    ----------
    encoder_layers : list of int
        The layer sizes for the VAE's encoder. The size of the first layer
        should be the same as the input size, while the size of the last
        layer should be the same as the size of the latent dimension.
    latent_dim : int
        The size of the latent space. 
    decoder_layers : list of int
        The layer sizes for the VAE's decoder. Should be the reverse of the
        encoder layers.
    """ 

    def __init__(self, encoder, decoder, cross_val):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cross_val = cross_val

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class LossFunction(nn.Module):

    def __init__(self):
        super(LossFunction, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        mse_loss = self.mse(x_recon, x)
        return mse_loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    