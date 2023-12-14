import time

import torch

from gather import GCN
from gather import GCN_DGL

from dgl.data import CoraGraphDataset
from dgl.utils import expand_as_pair
from dgl import function as fn

graph = CoraGraphDataset()
graph = graph[0]

input_dense = graph.ndata["feat"]

feat_src, feat_dst = expand_as_pair(input_dense, graph)
degs = graph.out_degrees().to(feat_src).clamp(min=1)
norm = torch.pow(degs, -0.5)
graph.srcdata.update({"di": norm})
graph.dstdata.update({"do": norm})
graph.apply_edges(fn.u_mul_v("di", "do", "dd"))

# print(graph.adj_tensors('csr')[0], graph.adj_tensors('csr')[0].shape)
edges = graph.number_of_edges()

in_feats = 32
out_feats = 128
iters = 100

# # TODO eval using a DGL graph and have a method to call DGL functions
# offsets = torch.arange(0, edges, edges_per_node, dtype=torch.int64)
# cols = torch.randint(0, nodes, (edges,), dtype=torch.int32)
# vals = torch.randn(edges, dtype=torch.float32)

offsets = graph.adj_tensors('csr')[0].to(torch.int64)
cols = graph.adj_tensors('csr')[1].to(torch.int32)
vals = torch.randn(edges, dtype=torch.float32)

gnn = GCN(in_feats, out_feats)
forward = 0
backward = 0
for _ in range(iters):
    start = time.time()
    new_h = gnn(input_dense, offsets, cols, vals)
    forward += time.time() - start

    start = time.time()
    backward += time.time() - start
print('Forward: {:.3f} s | Backward {:.3f} s'.format(forward, backward))


gnn = GCN_DGL(in_feats, out_feats)
forward = 0
backward = 0
for _ in range(iters):
    start = time.time()
    new_h = gnn(graph, input_dense)
    forward += time.time() - start

    start = time.time()
    backward += time.time() - start
print('Forward: {:.3f} s | Backward {:.3f} s'.format(forward, backward))