#imports
import os
import sys
import math
import random
from multiprocessing import Pool

import numpy as np
from scipy.spatial.distance import cdist, squareform
from scipy.cluster import hierarchy
import treeswift

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
        self.sigmoid = nn.Sigmoid()
        
        # decoder layers
        layers = zip(decoder_layers, decoder_layers[1:])
        layers = (self._add_layer(D_in, D_out) for D_in, D_out in layers)
        self.decoder = nn.Sequential(*layers)
        del self.decoder[-1][-1]
        
    def _add_layer(self, D_in, D_out):
        layers = (nn.Linear(D_in, D_out),
        nn.BatchNorm1d(num_features=D_out),
        nn.Sigmoid())
        return nn.Sequential(*layers)
        
    def encode(self, x):
        fc1 = F.sigmoid(self.latent(self.encoder(x)))
        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        return r1, r2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        fc3 = self.sigmoid(self.fc3(z))
        fc4 = self.sigmoid(self.fc4(fc3))
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
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    
    
    def resample(self, n_samples):
        """
        Resamples the fixed size embeddings (self.X) using the VAE.
        
        Parameters
        ----------
        n_samples : int
            number of samples to generate.
        
        Returns
        -------
        output : np.ndarray
            for a VAE trained on a dataset of fixed size embeddings of size (n, e)
            where (n) is each example and (e) is the fixed size embedding of each
            example. returns a array of resampled embeddings of size (s, n, e) where
            (s) is the n_samples specified in the input parameters.
        """
        torch.cuda.empty_cache()
        torch.set_num_threads(self.threads)
        self.model.eval()
        X_r = np.zeros((n_samples,*self.X.shape))
        with torch.no_grad():
            for index in range(X_r.shape[0]):
                data = next(iter(self.dataloader))
                data = data.to(self.device)
                output, mu, logvar = self.model(data)
                X_r[index] = output.cpu()
        return X_r
    
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
    

    
    
    
class distance_metric(torch.utils.data.Dataset):
    
    def __init__(self, embedding):
        self.embedding = embedding
        
    
    def get_metric(self, embedding, metric):
        if metric == "cosine":
            self.distmat = Metrics.cosine(embedding, embedding)
        if metric == "euclidean":
            self.distmat = Metrics.euclidean(embedding, embedding)
        if metric == "manhattan":
            self.distmat = Metrics.manhattan(embedding, embedding)
        if metric == "ts_ss":
            self.distmat = Metrics.ts_ss(embedding, embedding)
       
        return self.distmat
    
    

class neighbor_joining(torch.utils.data.Dataset):
    """
    Builds a tree from a distance matrix using the NJ algorithm using the
    original algorithm published by Saitou and Nei.

    Parameters
    ----------
    distmat : np.ndarray
        a square, symmetrical distance matrix of size (n, n)
    names : list of str
        list of size (n) containing names corresponding to the distance matrix

    Returns
    -------
    tree : str
        a newick-formatted tree
    """
    def __init__(self, distmat, headers):
        self.distmat = distmat
        self.headers = headers
        
        
    def join_ndx(D, n):
        # calculate the Q matrix and find the pair to join
        Q  = np.zeros((n, n))
        Q += D.sum(1)
        Q += Q.T
        Q *= -1.
        Q += (n - 2.) * D
        np.fill_diagonal(Q, 1.) # prevent from choosing the diagonal
        return np.unravel_index(Q.argmin(), Q.shape)

    def branch_lengths(D, n, i, j):
        i_to_j = float(D[i, j])
        i_to_u = float((.5 * i_to_j) + ((D[i].sum() - D[j].sum()) / (2. * (n - 2.))))
        if i_to_u < 0.:
            i_to_u = 0.
        j_to_u = i_to_j - i_to_u
        if j_to_u < 0.:
            j_to_u = 0.
        return i_to_u, j_to_u

    def update_distance(D, n1, mask, i, j):
        D1 = np.zeros((n1, n1))
        D1[0, 1:] = 0.5 * (D[i,mask] + D[j,mask] - D[i,j])
        D1[0, 1:][D1[0, 1:] < 0] = 0
        D1[1:, 0] = D1[0, 1:]
        D1[1:, 1:] = D[:,mask][mask]
        return D1
    
    def get_newick(self, distmat, headers):

        t = headers
        D = distmat.copy()
        np.fill_diagonal(D, 0.)

        while True:
            n = D.shape[0]
            if n == 3:
                break
            ndx1, ndx2 = neighbor_joining.join_ndx(D, n)
            len1, len2 = neighbor_joining.branch_lengths(D, n, ndx1, ndx2)
            mask  = np.full(n, True, dtype=bool)
            mask[[ndx1, ndx2]] = False
            t = [f"({t[ndx1]}:{len1:.6f},{t[ndx2]}:{len2:.6f})"] + [i for b, i in zip(mask, t) if b]
            D = neighbor_joining.update_distance(D, n-1, mask, ndx1, ndx2)

        len1, len2 = neighbor_joining.branch_lengths(D, n, 1, 2)
        len0 = 0.5 * (D[1,0] + D[2,0] - D[1,2])
        if len0 < 0:
            len0 = 0
        self.newick = f'({t[1]}:{len1:.6f},{t[0]}:{len0:.6f},{t[2]}:{len2:.6f});'
        return self.newick
    

class Metrics:
    """
    Functions for calculating distance matrices. I put them inside of a class to keep
    them organized in one place.

    Given two arrays of size (x, e) and (y, e), calculates a distance matrix
    of size (x, y).

    Example:
    >> A = np.random.random((100,1028))
    >> B = np.random.random((200,1028))
    >>
    >> Metrics.ts_ss(A, B)
    """
    @staticmethod
    def cosine(x1, x2):
        """
        >> from scipy.spatial.distance import cdist
        >> cdist(x1, x2, metric='cosine')
        """
        return cdist(x1, x2, metric='cosine')
    
    @staticmethod
    def euclidean(x1, x2):
        """
        >> from scipy.spatial.distance import cdist
        >> cdist(x1, x2, metric='euclidean')
        """
        return cdist(x1, x2, metric='euclidean')
    
    @staticmethod
    def manhattan(x1, x2):
        """
        >> from scipy.spatial.distance import cdist
        >> cdist(x1, x2, metric='cityblock')
        """
        return cdist(x1, x2, metric='cityblock')

    @staticmethod
    def jensenshannon(x1, x2):
        """
        >> from scipy.spatial.distance import cdist
        >> from scipy.special import softmax
        >> cdist(softmax(x1), softmax(x2), metric='jensenshannon')
        """
        x1 = softmax(x1, axis=1) # used to remove negative values
        x2 = softmax(x2, axis=1)
        return cdist(x1, x2, metric='jensenshannon')
    
    @staticmethod
    def ts_ss(x1, x2):
        """
        Stands for triangle area similarity (TS) and sector area similarity (SS)
        For more information: https://github.com/taki0112/Vector_Similarity
        """
        x1_norm = np.linalg.norm(x1, axis=-1)[:,np.newaxis]
        x2_norm = np.linalg.norm(x2, axis=-1)[:,np.newaxis]
        x_dot = x1_norm @ x2_norm.T

        ### cosine similarity
        cosine_sim = 1 - cdist(x1, x2, metric='cosine')
        cosine_sim[cosine_sim != cosine_sim] = 0
        cosine_sim = np.clip(cosine_sim, -1, 1, out=cosine_sim)

        ### euclidean_distance
        euclidean_dist = cdist(x1, x2, metric='euclidean')

        ### triangle_area_similarity
        theta = np.arccos(cosine_sim) + np.radians(10)
        triangle_similarity = (x_dot * np.abs(np.sin(theta))) / 2

        ### sectors area similarity
        magnitude_diff = np.abs(x1_norm - x2_norm.T)
        ed_plus_md = euclidean_dist + magnitude_diff
        sector_similarity =  ed_plus_md * ed_plus_md * theta * np.pi / 360

        ### hybridize
        similarity = triangle_similarity * sector_similarity
        return similarity
    

def cophenetic_distmat(tree, names):
    """
    Calculates the all-versus-all distance matrix of a tree based on the
    cophenetic distances.
    
    Parameters
    ----------
    tree : str
        a newick-formatted tree
    names : list of str
        a list of names contained within the tree. the order of the names provided
        in the list will be used to determine the order of the output.
    
    Returns
    -------
    cophmat : np.ndarray
        a square, symmetrical distance matrix
    """
    tree = treeswift.read_tree_newick(tree) if type(tree) is str else tree
    cophdic = tree.distance_matrix()
    node2name = {i:i.get_label() for i in cophdic.keys()}
    unique_names = set(node2name.values())
    assert len(node2name)==len(set(node2name))
    cophdic = {node2name[k1]:{node2name[k2]:cophdic[k1][k2] for k2 in cophdic[k1]} for k1 in cophdic.keys()}
    assert all(i in unique_names for i in names)
    cophmat = np.zeros([len(names)]*2)
    for ni, i in enumerate(names):
        for nj, j in enumerate(names[:ni]):
            cophmat[ni][nj] = cophmat[nj][ni] = cophdic[i][j]
    return cophmat 



def upgma(distmat, names):
    """
    Builds a tree from a distance matrix using the UPGMA algorithm.
    
    Parameters
    ----------
    distmat : np.ndarray
        a square, symmetrical distance matrix of size (n, n)
    names : list of str
        list of size (n) containing names corresponding to the distance matrix
    
    Returns
    -------
    tree : str
        a newick-formatted tree
    """
    D = squareform(distmat, checks=False)
    Z = hierarchy.linkage(D, method='average', optimal_ordering=False)
    def to_newick(node, newick, parentdist, leaf_names):
        if node.is_leaf():
            return f'{leaf_names[node.id]}:{parentdist-node.dist:.6f}{newick}'
        else:
            if len(newick) > 0:
                newick = f'):{parentdist - node.dist:.6f}{newick}'
            else:
                newick = ');'
            newick = to_newick(node.get_left(), newick, node.dist, leaf_names)
            newick = to_newick(node.get_right(), f',{newick}', node.dist, leaf_names)
            newick = f'({newick}'
            return newick
    tree = hierarchy.to_tree(Z, False)
    return to_newick(tree, "", tree.dist, names)



def get_bipartitions(tree, names):
    """
    Identify all bipartitions within a tree.
    
    Parameters
    ----------
    tree : ete3.Tree
        a tree object
    names : list of str
        a list of names contained within the tree. the order of the names provided
        in the list will be used to determine the order of the output.
    
    Returns
    -------
    bits : np.ndarray
        bipartition array of size (c, n) where c is each clade with more than one
        member on the tree and (n) is all taxa names on the tree. a bipartition
        for a given clade is represented based on which name is in the clade or not,
        0 if present and 1 if not present
    nodes : np.ndarray of ete3.Tree
        nodes corresponding to the (c) bipartitions
    """
    tips = set(i.name for i in tree.get_leaves())
    assert all(i in tips for i in names)
    
    n2i = {i: n for n, i in enumerate(names)}
    generator  = (n for n in tree.traverse("postorder") if not n.is_leaf())
    
    size  = len(names)
    bits  = np.zeros((size, size), dtype=np.int)
    nodes = np.zeros((size), dtype=object)
    
    for n, node in enumerate(generator):
        mask = [n2i[i.name] for i in node.get_leaves() if i.name in n2i]
        bits[n, mask] = 1
        if bits[n, 0] == 1:
            bits[n] = 1 - bits[n]
        nodes[n] = node
    
    bitsum  = bits.sum(1)
    include = (bitsum > 1) * (bitsum < size-1)
    
    return bits[include], nodes[include]



def get_support(reference_bits, sample_tree, names):
    """
    Get the branch support of a reference tree, given its bipartitions.
    
    Parameters
    ----------
    reference_bits : np.ndarray
        bipartition array of size (c, n) where c is each clade with more than one
        member on the tree and (n) is all taxa names on the tree. a bipartition
        for a given clade is represented based on which name is in the clade or not,
        0 if present and 1 if not present
    sample_tree : ete3.Tree
        a replicate tree
    names : list of str
        a list of names contained within the tree. the order of the names provided
        in the list will be used to map to the bipartitions.
    
    Returns
    -------
    out : np.ndarray
        for each reference bipartition, 1 if the bipartition is in the sample and
        0 if the bipartition is not
    """
    sample_bits = get_bipartitions(sample_tree, names)[0]
    return cdist(reference_bits, sample_bits, metric='cityblock').min(1) == 0

