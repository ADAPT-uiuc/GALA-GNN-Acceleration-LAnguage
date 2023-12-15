import time

import torch

from gather import GCN
from gather import GCN_DGL

from dgl.data import CoraGraphDataset, RedditDataset
from dgl.utils import expand_as_pair
from dgl import function as fn
import dgl

# graph = CoraGraphDataset()
graph = RedditDataset()
graph = graph[0]

# # Source nodes for edges (2, 1), (3, 2), (4, 3)
# src_ids = torch.tensor([0, 1, 2, 3, 4])
# # Destination nodes for edges (2, 1), (3, 2), (4, 3)
# dst_ids = torch.tensor([1, 2, 3, 4, 0])
# graph = dgl.graph((src_ids, dst_ids), num_nodes=5)
# graph.ndata["feat"] = torch.ones((5, 8))

input_dense = graph.ndata["feat"]
in_feats = input_dense.shape[1]
nrows = input_dense.shape[0]

feat_src, feat_dst = expand_as_pair(input_dense, graph)
degs = graph.out_degrees().to(feat_src).clamp(min=1)
# degs = torch.ones(nrows)
norm = torch.pow(degs, -0.5)
graph.srcdata.update({"di": norm})
graph.dstdata.update({"do": norm})
graph.apply_edges(fn.u_mul_v("di", "do", "dd"))

# print(graph.adj_tensors('csr')[0], graph.adj_tensors('csr')[0].shape)
edges = graph.number_of_edges()

out_feats = 32
iters = 100

# # TODO eval using a DGL graph and have a method to call DGL functions
# offsets = torch.arange(0, edges, edges_per_node, dtype=torch.int64)
# cols = torch.randint(0, nodes, (edges,), dtype=torch.int32)
# vals = torch.randn(edges, dtype=torch.float32)

offsets = graph.adj_tensors('csr')[0].to(torch.int64)
cols = graph.adj_tensors('csr')[1].to(torch.int32)
vals = graph.edata['dd']

gnn = GCN(in_feats, out_feats)
forward = 0
backward = 0
for _ in range(iters):
    start = time.time()
    new_h_cpp = gnn(input_dense, offsets, cols, vals)
    new_h_cpp = new_h_cpp[0]
    forward += time.time() - start

    start = time.time()
    backward += time.time() - start
print('Forward: {:.3f} s | Backward {:.3f} s'.format(forward, backward))

gnn = GCN_DGL(in_feats, out_feats)
forward = 0
backward = 0
for _ in range(iters):
    start = time.time()
    new_h_dgl = gnn(graph, input_dense)
    forward += time.time() - start

    start = time.time()
    backward += time.time() - start
print('Forward: {:.3f} s | Backward {:.3f} s'.format(forward, backward))

# print(new_h_cpp.shape)
# print(new_h_cpp)

# print(new_h_dgl.shape)
# print(new_h_dgl)

print(torch.isclose(new_h_cpp, new_h_dgl).sum() / (in_feats * nrows))
