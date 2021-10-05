from pynndescent import NNDescent
# from utils import lovasz_losses as L
from utils import utils

import torch
from torch import nn
from torch import optim
import numpy as np
import networkx as nx

from sklearn.neighbors import NearestNeighbors

import random

import criteria as C

def stress(pos, D, W, sampleSize=None):
    return C.stress(pos, D, W, sampleSize, reduce='mean').item()


def edge_uniformity(pos, G, k2i, sampleSize=None):
    return C.edge_uniformity(pos, G, k2i, sampleSize).item()


def crossings(pos, edge_indices):
    return utils.count_crossings(pos, edge_indices)


def neighborhood_preservation(pos, G, adj, i2k, lower=False):
    ## todo try sklearn/scipy knn?
    # from sklearn.neighbors import kneighbors_graph
    # kneighbors_graph(pos.detach().numpy(), 5, mode='distance', include_self=False).toarray()
    # # todo mask out less-than-node-degree nodes
    n,m = pos.shape
    
    ## knn
    degrees = [G.degree(i2k[i]) for i in range(len(G))]
    max_degree = max(degrees)
    
#     n_neighbors = max(2, min(max_degree+1, n))
#     n_trees = min(64, 5 + int(round((n) ** 0.5 / 20.0)))
#     n_iters = max(5, int(round(np.log2(n))))

#     knn_search_index = NNDescent(
#         pos.detach().numpy(),
#         n_neighbors=n_neighbors,
#         n_trees=n_trees,
#         n_iters=n_iters,
#         max_candidates=60,
#     )
#     knn_indices, knn_dists = knn_search_index.neighbor_graph
#     knn = np.zeros_like(adj)
#     for i in range(len(G)):
#         for j in range(degrees[i]):
#             knn[i, knn_indices[i,j+1]] = 1

    nn = NearestNeighbors(n_neighbors=max_degree+1)
    nn.fit(pos.detach())
    knn_indices = nn.kneighbors(pos.detach())[1]
    knn = np.zeros_like(adj)
    for i in range(len(G)):
        for j in range(degrees[i]):
            knn[i, knn_indices[i,j+1]] = 1
    jaccard = (np.logical_and(adj,knn).sum() / np.logical_or(adj,knn).sum()).item()
    if lower:
        return 1-jaccard
    else:
        return jaccard



def crossing_angle_maximization(pos, G_edges, k2i):
    quality = 0
    crossings = utils.find_crossings(pos, G_edges, k2i)
    if len(crossings) > 0:
        pos_segs = pos[crossings.flatten()].view(-1,4,2)
        v1 = pos_segs[:,1] - pos_segs[:,0]
        v2 = pos_segs[:,3] - pos_segs[:,2]
        cosSim = torch.nn.CosineSimilarity()(v1, v2)
        cosSim.clamp_(-1,1)
        angle = torch.acos(cosSim)
        
        ##quality is the lower the better
        quality = (angle - np.pi/2).abs().max().item() / (np.pi/2) 
#         quality = (angle - np.pi/2).abs().mean().item() / (np.pi/2)
    return quality



def aspect_ratio(
    pos, sampleSize=None, 
    rotation_angles=torch.arange(7,dtype=torch.float)/7*(np.pi/2), 
    lower=False
):
    if sampleSize is not None:
        n = pos.shape[0]
        i = np.random.choice(n, min(n,sampleSize), replace=False)
        samples = pos[i,:]
    else:
        samples = pos.clone()
    mean = samples.mean(dim=0, keepdim=True)
    samples -= mean
    
    cos = torch.cos(rotation_angles)
    sin = torch.sin(rotation_angles)
    rot = torch.stack([cos, sin, -sin, cos], 1).view(len(rotation_angles), 2, 2)
    
    samples = samples.matmul(rot)
    
    wh = samples.max(1).values - samples.min(1).values
    ratios = wh.min(1).values / wh.max(1).values 
    quality = ratios.min().item() ## range=[0,1] thecloser to 1 the better
    if lower:
        return 1-quality
    else:
        return quality


def angular_resolution(pos, G, k2i, sampleSize=None):
    if sampleSize is None:
        samples = G.nodes
    else:
        samples = utils.sample_nodes(G, sampleSize)
    neighbors = [list(G.neighbors(s)) for s in samples]
    max_degree = len(max(neighbors, key=lambda x:len(x)))
    sampleIndices = [k2i[s] for s in samples]
    neighborIndices = [[k2i[n] for n in nei] for nei in neighbors]
    
    samples = pos[sampleIndices]
    neighbors = [pos[nei] for nei in neighborIndices]
    
    rays = [nei-sam for nei,sam in zip(neighbors, samples) if len(nei)>1]
    angles = [utils.get_angles(rs) for rs in rays]
    min_angle = min(min(a) for a in angles)
    ar = min_angle / (np.pi*2/max_degree)
    return ar.item()



def vertex_resolution(pos, sampleSize=None, target=0.1, lower=False):
    pairwiseDistance = nn.PairwiseDistance()
    n = pos.shape[0]
    if sampleSize is not None:
        i = np.random.choice(n, min(n,sampleSize), replace=False)
        samples = pos[i,:]
    else:
        samples = pos
    m = samples.shape[0]
    a = samples.repeat([1,m]).view(-1,2)
    b = samples.repeat([m,1])
    pdist = pairwiseDistance(a, b)
    dmax = pdist.max().detach()
    pdist[::m+1] = 1e9
    vr = (pdist.min() / (dmax*target)).item()
    if lower:
        return 1-vr
    else:
        return vr



def gabriel(pos, G, k2i, sampleSize=None, lower=False):

    if sampleSize is None:
        nodes = G.nodes
        edges = G.edges
    else:
        edges = utils.sample_edges(G, sampleSize)
        nodes = utils.sample_nodes(G, sampleSize)
    
    edges = np.array([(k2i[e0], k2i[e1]) for e0,e1 in edges])
    nodes = np.array([k2i[n] for n in nodes])
    node_pos = pos[nodes]
    edge_pos = pos[edges.flatten()].reshape([-1,2,2])
    centers = edge_pos.mean(1)
    radii = (edge_pos[:,0,:] - edge_pos[:,1,:]).norm(dim=1)/2
    node_center_dists = (node_pos.reshape([-1,1,2])-centers.reshape([1,-1,2])).norm(dim=-1)
    node_center_dists /= radii
    
    quality = node_center_dists.min().item()
    if lower:
        return 1-quality
    else:
        return quality
        
        
        
        
        
        
        
        
        
        