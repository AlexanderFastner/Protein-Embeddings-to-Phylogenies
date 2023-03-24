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
    def __init__(self, encoder_layers, latent_dim, decoder_layers):
        super(VariationalAutoencoder, self).__init__()
    
        # encoder layers
        layers = zip(encoder_layers, encoder_layers[1:])
        layers = (self._add_layer(D_in, D_out) for D_in, D_out in layers)
        self.encoder = nn.Sequential(*layers)
        
        # latent vectors mu & sigma
        self.latent = self._add_layer(encoder_layers[-1], latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)
        
        # sampling vector epsilon
        self.fc3 = self._add_layer(latent_dim, latent_dim)
        self.fc4 = self._add_layer(latent_dim, decoder_layers[0])
        self.relu = nn.ReLU()
        
        # decoder layers
        layers = zip(decoder_layers, decoder_layers[1:])
        layers = (self._add_layer(D_in, D_out) for D_in, D_out in layers)
        self.decoder = nn.Sequential(*layers)
        del self.decoder[-1][-1]
    
    def _add_layer(self, D_in, D_out):
        layers = (nn.Linear(D_in, D_out),
            nn.BatchNorm1d(num_features=D_out),
            nn.ReLU())
        return nn.Sequential(*layers)
        
    def encode(self, x):
        fc1 = F.relu(self.latent(self.encoder(x)))
        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        return r1, r2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        fc3 = self.relu(self.fc3(z))
        fc4 = self.relu(self.fc4(fc3))
        return self.decoder(fc4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    #TODO add cross val
    


class LossFunction(nn.Module):

    def __init__(self):
        super(LossFunction, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        mse_loss = self.mse(x_recon, x)
        return mse_loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    