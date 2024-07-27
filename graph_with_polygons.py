# graph_with_polygons.py

import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torch import nn
from typing import List, Tuple

class GraphWithPolygons:
    def __init__(self, edges: List[Tuple[str, str]], size_factor: float = 1.0):
        self.graph = nx.Graph()
        self.graph.add_edges_from(edges)
        self.size_factor = size_factor
        self.radii = self._initialize_radii()
        self.node_mapping = {node: idx for idx, node in enumerate(self.graph.nodes)}

    def _initialize_radii(self) -> torch.Tensor:
        label_lengths = torch.tensor([len(node) for node in self.graph.nodes])
        radii = self.size_factor * (0.02 * label_lengths + 0.1)
        return radii

def create_polygon(x: float, y: float, radius: float, n_sides: int) -> Polygon:
    theta = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    vertices = np.vstack((np.cos(theta), np.sin(theta))).T * radius + np.array([x, y])
    return Polygon(vertices, closed=True, color='lightblue', ec='black')

def generate_random_points(num_points: int) -> torch.Tensor:
    return torch.rand(size=(num_points, 2), requires_grad=True)

def draw_graph_with_polygons(graph: GraphWithPolygons, points: torch.Tensor, n_sides: int, title: str):
    fig, ax = plt.subplots()
    pos = {node: (points[i][0].item(), points[i][1].item()) for i, node in enumerate(graph.graph.nodes())}

    for node, (x, y) in pos.items():
        radius = graph.radii[list(graph.graph.nodes()).index(node)].item()
        polygon = create_polygon(x, y, radius, n_sides)
        ax.add_patch(polygon)
        ax.text(x, y, node, ha='center', va='center', fontsize=12)

    for edge in graph.graph.edges():
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        ax.plot([x1, x2], [y1, y2], color='gray')

    plt.title(title)
    plt.axis('equal')
    plt.axis('off')
    plt.show()

def node_overlap_loss(pos: torch.Tensor, radii: torch.Tensor, sample_size=None, sample=None) -> torch.Tensor:
    pairwise_distance = nn.PairwiseDistance()
    relu = nn.ReLU()

    n = pos.shape[0]
    if sample is None:
        if sample_size is None or sample_size == 'full':
            indices = torch.arange(n)
        else:
            indices = torch.randperm(n)[:sample_size]
        sample = torch.cartesian_prod(indices, indices)

    a = pos[sample[:, 0]]
    b = pos[sample[:, 1]]
    radii_a = radii[sample[:, 0]]
    radii_b = radii[sample[:, 1]]
    pdist = pairwise_distance(a, b)

    normalized_dist = pdist / (radii_a + radii_b)
    loss = relu(1 - normalized_dist).pow(2).mean()

    return loss
