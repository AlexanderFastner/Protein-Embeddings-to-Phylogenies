#imports
import os
import sys
import math
import random
from multiprocessing import Pool

import numpy as np
from scipy.spatial.distance import cdist

import pytorch_lightning as pl

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
class VariationalAutoencoder(pl.LightningModule):
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

    def kl_divergence_loss(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
        return kl_loss
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        out, mu, logvar = self.forward(x)
        recon_loss = self.reconstruction_loss(x, out)
        kl_loss = self.kl_divergence_loss(mu, logvar)
        loss = recon_loss + kl_loss
        logs = {'reconstruction_loss': recon_loss, 'kl_loss': kl_loss}
        values = {"loss": loss, 'reconstruction_loss': recon_loss, 'kl_loss': kl_loss}
        self.log_dict(values)
        return {'loss': loss, 'log': logs}    
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        out, mu, logvar = self.forward(x)
        recon_loss = self.reconstruction_loss(x, out)
        kl_loss = self.kl_divergence_loss(mu, logvar)
        loss = recon_loss + kl_loss
        logs = {'reconstruction_loss': recon_loss, 'kl_loss': kl_loss}
        val_loss = {"validation_loss": loss}
        self.log_dict(val_loss)
        return {'log': logs}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def reconstruction_loss(self, x, out):
        recon_loss = F.mse_loss(x, out)
        return recon_loss
    
class LossFunction(pl.LightningModule):
    
    def __init__(self, embeddings_dim):
        super().__init__()
        self.embeddings_dim = embeddings_dim

    def forward(self, x, recon_x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    
class makedataset(torch.utils.data.Dataset):

    def __init__(self, headers, embeddings):
        self.headers = headers
        self.embeddings = embeddings
        assert len(self.headers) == len(self.embeddings)
        self.data = list(zip(embeddings, headers))

    def __len__(self):
        return len(self.data)

    #need a map of dataset[index] to return that embedding and its header
    #each header and embeddinng is entered into a list and gets an index
    def __getitem__(self, index):
        data = self.data[index]
        #print("_getitem_", index)
        #print("self.data[index]", self.data[index])
        return data
    
    
    
    
    
    
    
    
    
    
    
    