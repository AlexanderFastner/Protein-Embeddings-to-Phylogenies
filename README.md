# Protein Embeddings to Phylogenies

This is an exploration of using dimensionality reduction to create Phylogenetic trees faster than traditional methods with Multiple Sequence Alignments.

## by Adel Schmucklermann, Alexander Fastner

## Abstract
We explored the use of protein sequence embeddings generated with ProtT5 and Esm-2 to construct phylogenetic trees, aiming to reduce the time required compared to traditional methods. To filter out the noise and extract more information, we employed PCA and t-SNE, as well as a Variational Auto-Encoder (VAE), to reduce embedding dimensions. We then calculate various distance metrics and cluster them into trees using UPGMA and Neighbor-Joining.

<img src="/test/Screenshot 2023-07-04 132908.png" alt="Workflow" title="Basic workflow">

### [Paper](./Embeddings_to_Phylogenies.pdf)

## Run Script

To run the notebook yourself create a conda env using the provioded [Environment File](./environment.yaml)

### [Main script](./VAE.ipynb)
