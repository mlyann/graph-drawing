import networkx as nx
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from collections import defaultdict
import random
import time
import os
import itertools
import pickle as pkl
from utils import utils, vis
import criteria as C
import quality as Q  # Make sure to import the updated quality module
from gd2 import GD2
import utils.weight_schedule as ws
from graph_with_polygons import GraphWithPolygons, generate_random_points, draw_graph_with_polygons

from typing import List, Tuple

# Define the device and seed for reproducibility
device = 'cpu'
seed = 2337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Define the function to read edges from a file
def read_edges_from_file(file_path: str) -> List[Tuple[str, str]]:
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(" -- ")
            edge = (parts[0].strip('"'), parts[1].strip('"'))
            edges.append(edge)
    return edges

# File path within the cloned repository
file_path = 'multi_level_tree/dataset/lastfm_refined/Graph_3_800.txt'

# Read the edges from the file
edges = read_edges_from_file(file_path)

# Create the graph with polygons
graph_with_polygons = GraphWithPolygons(edges, size_factor=1.0)

# Generate random points for the initial positions
points = generate_random_points(len(graph_with_polygons.graph.nodes))

# Calculate and print initial stress and node overlap
initial_stress = Q.stress(points, graph_with_polygons.graph)
initial_overlap = Q.calculate_node_overlap(points, graph_with_polygons.radii)
print(f"Initial Stress: {initial_stress}")
print(f"Initial Node Overlap: {initial_overlap}")

draw_graph_with_polygons(graph_with_polygons, points, n_sides=6, title="Initial Random Layout")

# Parameters for the gradient descent algorithm
max_iter = int(1e4)
learning_rate = 0.01

# Define criteria weights and sample sizes
criteria = ['stress', 'vertex_resolution']
criteria_weights = dict(
    stress=ws.SmoothSteps([max_iter / 4, max_iter], [1, 0.05]),
    vertex_resolution=ws.SmoothSteps([0, max_iter * 0.2, max_iter * 0.6, max_iter], [0, 0, 0.5, 0]),
)
sample_sizes = dict(
    stress=16,
    vertex_resolution=max(256, int(len(graph_with_polygons.graph.nodes()) ** 0.5)),
)

# Initialize GD2 and optimize
gd = GD2(graph_with_polygons.graph)
result = gd.optimize(
    criteria_weights=criteria_weights,
    sample_sizes=sample_sizes,
    evaluate=set(criteria),
    max_iter=max_iter,
    time_limit=3600,  # sec
    evaluate_interval=max_iter,
    evaluate_interval_unit='iter',
    vis_interval=-1,
    vis_interval_unit='sec',
    clear_output=True,
    grad_clamp=20,
    criteria_kwargs=dict(
        vertex_resolution=dict(target=[1, 1]),
    ),
    optimizer_kwargs=dict(mode='SGD', lr=2),
    scheduler_kwargs=dict(verbose=True),
)

# Output the positions and plot the graph
pos = gd.pos.detach().numpy().tolist()
pos_G = {k: pos[gd.k2i[k]] for k in gd.G.nodes}

print('nodes')
for node_id, pos in pos_G.items():
    print(f'{node_id}, {pos[0]}, {pos[1]}')

print('edges')
for e in gd.G.edges:
    print(f'{e[0]}, {e[1]}')

# 绘制优化后的布局
draw_graph_with_polygons(graph_with_polygons, gd.pos, n_sides=6, title="Optimized Layout")

# Calculate final stress and node overlap
final_stress = Q.stress(gd.pos, graph_with_polygons.graph)
final_overlap = Q.calculate_node_overlap(gd.pos, graph_with_polygons.radii)
print(f"Final Stress: {final_stress}")
print(f"Final Node Overlap: {final_overlap}")
