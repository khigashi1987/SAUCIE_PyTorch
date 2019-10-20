"""
PyTorch implementation of SAUCIE algorithm.
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch import nn
import itertools
import sklearn.metrics

CONFIG = { 
    'latent_size': 2, # embedding dimension.
    'lambda_b': 0.05, # regularization rate of MMD loss.
    'lambda_c': 0.5, # reg. rate of Entropy loss.
    'lambda_d': 0.7, # reg. rate of Intra-cluster distance loss.
    'binmin': 10, # minimum number of cells to be clustered.
    'max_clusters': 100, # max number of clusters.
    'merge_k_nearest': 3, # number of nearest clusteres to search in merging process.
    'layers': [512, 256, 128], # number of nodes in each layer of encoder and decoder.
    'learning_rate': 1e-3, # learning rate of optimizer.
    'minibatch_size': 256,
    'use_batchnorm': True, # use batch normalization layer.
    'max_iterations': 1000, # max iteration steps
    'log_interval': 100, # interval of steps to display loss information.
    'use_gpu': False, 
    'train_dir': './tmp', # dir to save the state of the model
    'data_dir': './data', # input files dir
    'out_dir': './results', # dir to output result files
    'seed':13 # seed of random number generators
}

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def asinh(x, scale=5.):
    f = np.vectorize(lambda y: math.asinh(y / scale))
    return f(x)

def sinh(x, scale=5.):
    f = np.vectorize(lambda y: scale * math.sinh(y))
    return f(x)

def pairwise_dist(x1, x2):
    r1 = torch.sum(x1*x1, 1, keepdim=True)
    r2 = torch.sum(x2*x2, 1, keepdim=True)
    K = r1 - 2*torch.matmul(x1, torch.t(x2)) + torch.t(r2)
    return K
    
def gaussian_kernel_matrix(dist):
    # Multi-scale RBF kernel. (average of some bandwidths of gaussian kernels)
    # This must be properly set for the scale of embedding space
    sigmas = [1e-5, 1e-4, 1e-3, 1e-2]
    beta = 1. / (2. * torch.unsqueeze(torch.tensor(sigmas), 1))
    s = torch.matmul(beta, torch.reshape(dist, (1, -1)))
    return torch.reshape(torch.sum(torch.exp(-s), 0), dist.shape) / len(sigmas)

def load_expression_table(DFs, cfg, **kwargs):
    # Load the list of pandas dataframes, label the batch IDs, and return the loader of data and labels.
    data = asinh(DFs[0].values)
    labels = [np.zeros(shape=(len(DFs[0]),1))]
    for i, df in enumerate(DFs[1:]):
        data = np.vstack((data, asinh(df.values)))
        labels.append(np.array([i+1]*len(df)).reshape((len(df), 1)))
    labels = np.concatenate(labels, axis=0)
    data_label = torch.utils.data.TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(labels).float())
    data_loader = torch.utils.data.DataLoader(data_label, batch_size=cfg['minibatch_size'], shuffle=True, drop_last=True, **kwargs)
    return data_loader

class Encoder(nn.Module):
    """
    Encoding layers.
    """
    def __init__(self, n_features, cfg):
        super().__init__()
        if cfg['use_batchnorm']:
            modules = [nn.Linear(n_features, cfg['layers'][0]), nn.BatchNorm1d(cfg['layers'][0]), nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(cfg['layers'][0], cfg['layers'][1]), nn.BatchNorm1d(cfg['layers'][1]), nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(cfg['layers'][1], cfg['layers'][2]), nn.BatchNorm1d(cfg['layers'][2]), nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(cfg['layers'][2], cfg['latent_size']), nn.Identity()]
        else:
            modules = [nn.Linear(n_features, cfg['layers'][0]), nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(cfg['layers'][0], cfg['layers'][1]), nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(cfg['layers'][1], cfg['layers'][2]), nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(cfg['layers'][2], cfg['latent_size']), nn.Identity()]
        self.net = nn.Sequential(*modules)
    def forward(self, input):
        # output embedding activations
        return self.net(input)

class Cluster(nn.Module):
    """
    Decoding and clustering layers.
    """
    def __init__(self, cfg):
        super().__init__()
        if cfg['use_batchnorm']:
            modules = [nn.Linear(cfg['latent_size'], cfg['layers'][2]), nn.BatchNorm1d(cfg['layers'][2]), nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(cfg['layers'][2], cfg['layers'][1]), nn.BatchNorm1d(cfg['layers'][1]), nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(cfg['layers'][1], cfg['layers'][0]), nn.BatchNorm1d(cfg['layers'][0]), nn.ReLU()]
        else:
            modules = [nn.Linear(cfg['latent_size'], cfg['layers'][2]), nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(cfg['layers'][2], cfg['layers'][1]), nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(cfg['layers'][1], cfg['layers'][0]), nn.ReLU()]
        self.net = nn.Sequential(*modules)
    def forward(self, input):
        # output clustering activations
        return self.net(input)

class Decoder(nn.Module):
    """
    Reconstructing layer.
    """
    def __init__(self, n_features, cfg):
        super().__init__()
        modules = [nn.Linear(cfg['layers'][0], n_features)]
        self.net = nn.Sequential(*modules)
    def forward(self, input):
        # output reconstructing activations
        return self.net(input)

def loss_reconstruction(recon, x, labels):
    """
    Reconstruction loss part of the network.
    """
    labels = labels.reshape(-1)
    loss = 0
    n_ref = (labels == 0).sum()
    # reference batch: Reconstruction loss
    if n_ref > 0:
        ref_x = x[labels == 0, :]
        ref_recon = recon[labels == 0, :]
        mseloss = nn.MSELoss()
        loss = mseloss(ref_x, ref_recon)
    # non-reference batches: comparing normalized distribution only
    for nonref_batch_ind in range(int(labels.max().detach().numpy())+1)[1:]:
        n_nonref = (labels == nonref_batch_ind).sum()
        if n_nonref > 0:
            nonref_x = x[labels == nonref_batch_ind, :]
            mean_x = torch.mean(nonref_x, 0)
            var_x = torch.var(nonref_x, 0)
            nonref_recon = recon[labels == nonref_batch_ind, :]
            mean_recon = torch.mean(nonref_recon, 0)
            var_recon = torch.var(nonref_recon, 0)
            loss += n_nonref.float() / float(len(labels)) * \
                mseloss(torch.div(torch.add(nonref_recon, -1.0 * mean_recon), (torch.sqrt(var_recon+1e-12)+1e-12)), \
                    torch.div(torch.add(nonref_x, -1.0 * mean_x), (torch.sqrt(var_x+1e-12)+1e-12))) 
    return loss

def reg_b(embed, labels):
    """
    Maximum Mean Discrepancy regularization.
    """
    labels = labels.reshape(-1)
    n_others = torch.unique(labels).size()[0] - 1
    e = embed / torch.mean(embed)
    K = pairwise_dist(e, e)
    K = K / torch.max(K)
    K = gaussian_kernel_matrix(K)
    loss = 0
    # reference batch (batch_ind == 0) vs. other batches
    # single-batch term in MMD
    for batch_ind in range(int(labels.max().detach().numpy()) + 1):
        batch_rows = K[labels == batch_ind, :]
        batch_K = torch.t(batch_rows)[labels == batch_ind, :]
        batch_nrows = torch.sum(torch.ones_like(labels)[labels == batch_ind]).float()
        var_within_batch = torch.sum(batch_K) / (batch_nrows**2)
        if batch_ind == 0:
            loss += var_within_batch * n_others
        else:
            loss += var_within_batch
    # between-batches (reference vs. other batch) term in MMD
    nrows_ref = torch.sum(torch.ones_like(labels)[labels == 0]).float()
    for batch_ind_vs in range(int(labels.max().detach().numpy())+1)[1:]:
        K_bw = K[labels == 0, :]
        K_bw = torch.t(K_bw)[labels == batch_ind_vs, :]
        K_bw = torch.sum(torch.t(K_bw))
        nrows_vs = torch.sum(torch.ones_like(labels)[labels == batch_ind_vs]).float()
        loss -= (2*K_bw) / (nrows_ref * nrows_vs)
    return loss

def reg_c(act):
    """
    ID (information dimension) regularization (minimizing Shannon entropy)
    """
    p = torch.sum(act, 0, keepdim=True)
    normalized = p / torch.sum(p)
    return torch.sum(-normalized * torch.log(normalized + 1e-9))

def reg_d(act, x):
    """
    Intra-cluster distances regularization.
    """
    act = act / torch.max(act)
    dist = pairwise_dist(act, act)
    same_cluster_probs = gaussian_kernel_matrix(dist)
    same_cluster_probs = same_cluster_probs - torch.min(same_cluster_probs)
    same_cluster_probs = same_cluster_probs / torch.max(same_cluster_probs)
    original_dist =  pairwise_dist(x, x)
    original_dist = torch.sqrt(original_dist + 1e-3)
    intracluster_dist = torch.matmul(original_dist, same_cluster_probs)
    return torch.mean(intracluster_dist)

def get_cluster_merging(clusters, embeddings, merge_k_nearest):
    """
    Merge clusters based on bi-directional best hit using MMD distance between clusters.
    """
    if len(np.unique(clusters))==1: return clusters
    clusts_to_use = np.unique(clusters)
    mmdclusts = np.zeros((len(clusts_to_use), len(clusts_to_use)))
    for i1, i2 in itertools.combinations(range(len(clusts_to_use)), 2):
        clust1 = clusts_to_use[i1]
        clust2 = clusts_to_use[i2]
        if clust1 == -1 or clust2 == -1:
            continue
        ei = embeddings[clusters == clust1]
        ej = embeddings[clusters == clust2]
        ri = list(range(ei.shape[0])); np.random.shuffle(ri); ri = ri[:1000];
        rj = list(range(ej.shape[0])); np.random.shuffle(rj); rj = rj[:1000];
        ei = ei[ri, :]
        ej = ej[rj, :]
        k1 = sklearn.metrics.pairwise.pairwise_distances(ei, ei)
        k2 = sklearn.metrics.pairwise.pairwise_distances(ej, ej)
        k12 = sklearn.metrics.pairwise.pairwise_distances(ei, ej)
        mmd = 0
        for sigma in [.01, .1, 1., 10.]:
            k1_ = np.exp(- k1 / (sigma**2))
            k2_ = np.exp(- k2 / (sigma**2))
            k12_ = np.exp(- k12 / (sigma**2))
            mmd += k1_.sum()/(k1_.shape[0]*k1_.shape[1]) +\
                   k2_.sum()/(k2_.shape[0]*k2_.shape[1]) -\
                   2*k12_.sum()/(k12_.shape[0]*k12_.shape[1])
        mmdclusts[i1, i2] = mmd
        mmdclusts[i2, i1] = mmd

    clust_reassign = {}
    for i1, i2 in itertools.combinations(range(mmdclusts.shape[0]), 2):
        k5_1 = np.argsort(mmdclusts[i1, :])[1:merge_k_nearest+1]
        k5_2 = np.argsort(mmdclusts[i2, :])[1:merge_k_nearest+1]
        if np.isin(i2, k5_1) and np.isin(i1, k5_2):
            clust_reassign[clusts_to_use[i2]] = clusts_to_use[i1]
            clusts_to_use[i2] = clusts_to_use[i1]

    for c in clust_reassign:
        mask = clusters == c
        clusters[mask] = clust_reassign[c]
    return clusters

def get_clusters(acts, embedding_DFs, binmin=10, max_clusters=100, merge_k_nearest=1):
    """
    Clustering function. Binarize activations on clustering layer and aggregate binary codes.
    """
    acts = acts / acts.max()
    binarized = np.where(acts > 1e-6, 1, 0)
    unique_rows, counts = np.unique(binarized, axis=0, return_counts=True)
    unique_rows = unique_rows[counts > binmin]
    n_clusters = unique_rows.shape[0]

    if n_clusters > max_clusters:
        print("Too many clusters ({}) to go through...".format(n_clusters))
        return n_clusters, np.zeros(acts.shape[0])

    n_clusters = 0
    clusters = -1 * np.ones(acts.shape[0])
    for i, row in enumerate(unique_rows):
        rows_equal_to_this_code = np.where(np.all(binarized == row, axis=1))[0]

        clusters[rows_equal_to_this_code] = n_clusters
        n_clusters += 1

    print(f'{len(np.unique(clusters))} clusters detected. Merging clusters...')
    print(f'\t{(clusters == -1).astype(int).sum()} cells '
        f'({100. * (clusters == -1).astype(int).sum() / float(len(clusters)):.2f} % of total) are not clustered.')
    # merge clusters
    embeddings = np.vstack([df.values for df in embedding_DFs])
    clusters = get_cluster_merging(clusters, embeddings, merge_k_nearest)
    n_clusters = len(np.unique(clusters))
    print(f'Merging done. Total {len(np.unique(clusters))} clusters.')
    return n_clusters, clusters

def train(DFs, mode='BatchCorrection', cfg={}, device=torch.device('cpu')):
    """
    Train neural network.
    """
    n_features = len(DFs[0].columns)

    # initialize model
    encoder = Encoder(n_features, cfg)
    cluster = Cluster(cfg)
    decoder = Decoder(n_features, cfg)
    encoder.to(device)
    cluster.to(device)
    decoder.to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + \
        list(cluster.parameters()) + \
        list(decoder.parameters()), \
        lr=cfg['learning_rate'])

    kwargs = {'num_workers':4, 'pin_memory':True} if cfg['use_gpu'] else {}
    train_data = load_expression_table(DFs, cfg, **kwargs)

    print(f'Training the {mode} model for {len(DFs)} datasets')
    training_losses = []
    for step, batch in enumerate(cycle(train_data)):
        x = batch[0].to(device)
        encoder.zero_grad()
        cluster.zero_grad()
        decoder.zero_grad()

        embed = encoder(x)
        cl = cluster(embed)
        recon = decoder(cl)

        # reconstruction loss
        recon_loss = loss_reconstruction(recon, x, batch[1])

        if mode == 'BatchCorrection':
            # MMD regularization
            mmd_loss = reg_b(embed, batch[1])
            loss = recon_loss + cfg['lambda_b'] * mmd_loss
        if mode == 'Clustering':
            # ID regularization
            id_loss = reg_c(cl)
            # Intra-cluster distance regularization
            intradist_loss = reg_d(cl, x)
            loss = recon_loss + cfg['lambda_c'] * id_loss + cfg['lambda_d'] * intradist_loss
            #print('id_loss:', id_loss.detach().numpy(), ' intradist_loss:', intradist_loss.detach().numpy())
        loss.backward()
        optimizer.step()

        training_losses.append(loss.detach().cpu().numpy())
        if step % cfg['log_interval'] == 0:
            print(f'\tstep:\t{step}\ttrain loss: {np.array(training_losses).mean():.6f}')
            training_losses = []
        if step > cfg['max_iterations']:
            states = {'encoder': encoder.state_dict(),
                'cluster': cluster.state_dict(),
                'decoder': decoder.state_dict()}
            torch.save(states, os.path.join(cfg['train_dir'], f'state_dict'))
            break

def output_activations(DFs, mode='BatchCorrection', cfg={}):
    """
    Output activations of embedding layer, clustering layer, and reconstructing layer.
    """
    print('Reconstructing data by using trained model...')
    n_features = len(DFs[0].columns)
    results_embedded = []
    results_reconstructed = []
    all_cl_activations = []
    for df in DFs:
        encoder = Encoder(n_features, cfg)
        cluster = Cluster(cfg)
        decoder = Decoder(n_features, cfg)
        checkpoint = torch.load(os.path.join(cfg['train_dir'], f'state_dict'))
        encoder.load_state_dict(checkpoint['encoder'])
        cluster.load_state_dict(checkpoint['cluster'])
        decoder.load_state_dict(checkpoint['decoder'])
        encoder.eval()
        cluster.eval()
        decoder.eval()

        x = torch.tensor(asinh(df.values), dtype=torch.float)
        embed = encoder(x)
        cl = cluster(embed)
        recon = decoder(cl)

        results_embedded.append(pd.DataFrame(embed.detach().numpy(), index=df.index, \
            columns=[f'Dim{i+1}' for i in range(cfg['latent_size'])]))
        results_reconstructed.append(pd.DataFrame(sinh(recon.detach().numpy()), index=df.index, columns=df.columns))
        if mode == 'Clustering':
            all_cl_activations.append(cl.detach().numpy())

    if mode == 'BatchCorrection':
        return results_embedded, results_reconstructed
    elif mode == 'Clustering':
        all_cl_activations = np.vstack(all_cl_activations)
        n_clusters, clusters = get_clusters(all_cl_activations, results_embedded, \
            binmin=cfg['binmin'], max_clusters=cfg['max_clusters'], merge_k_nearest=cfg['merge_k_nearest'])
        split_indices = np.cumsum(np.array([len(df) for df in DFs]))[:-1]
        return np.split(clusters, split_indices)

if __name__ == '__main__':
    cfg = CONFIG
    device = torch.device("cuda:0" if cfg['use_gpu'] else "cpu")
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    DFs = []
    names = []
    for csv_file in glob.glob(os.path.join(cfg['data_dir'], '*.csv')):
        names.append(os.path.basename(csv_file).split('.')[0])
        DFs.append(pd.read_csv(csv_file, index_col=0))
    
    # train batch correction model
    train(DFs, mode='BatchCorrection', cfg=cfg, device=device)

    # Batch correction by using trained model
    results_embedded, results_reconstructed = output_activations(DFs, mode='BatchCorrection', cfg=cfg)

    # write files
    for (df, name) in zip(results_embedded, names):
        df.to_csv(os.path.join(cfg['out_dir'], f'{name}_embedding.csv'))
    for (df, name) in zip(results_reconstructed, names):
        df.to_csv(os.path.join(cfg['out_dir'], f'{name}_reconstructed.csv'))
    
    # train clustering model
    train(results_reconstructed, mode='Clustering', cfg=cfg, device=device)

    # get clusters and output
    results_clusters = output_activations(results_reconstructed, mode='Clustering', cfg=cfg)
    for (clusters, name) in zip(results_clusters, names):
        np.savetxt(os.path.join(cfg['out_dir'], f'{name}_clusters.txt'), clusters.astype(int), fmt='%d')