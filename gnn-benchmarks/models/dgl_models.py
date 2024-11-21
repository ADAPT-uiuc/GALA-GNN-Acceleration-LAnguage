import time
import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, GATConv, SAGEConv, GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling

'''
layerTimes has every forward layer time in array, size is number of layers
forwardTime is a float representing total foward propogation time
'''
class dgl_GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_size, hid_size[0], norm="both"))
        for i in range(len(hid_size)-1):
            self.layers.append(GraphConv(hid_size[i], hid_size[i+1], norm="both"))
        self.layers.append(GraphConv(hid_size[-1], out_size, norm="both"))

        

    def forward(self, graph, in_feat):
        # fix for arxiv dataset
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        
        start = time.time()
        layerTimes = []
        for i, layer in enumerate(self.layers):
            startLayer = time.time()
            h = layer(graph, in_feat if i == 0 else h)
            h = F.relu(h)
            layerTimes.append(time.time() - startLayer)

        return h, layerTimes, time.time() - start
    
class dgl_GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_size, hid_size[0], heads[0]))
        for i in range(len(hid_size)-1):
            self.layers.append(GATConv(hid_size[i]*heads[i], hid_size[i+1], heads[i+1]))
        self.layers.append(GATConv(hid_size[-1]*heads[-2], out_size, heads[-1]))
        
    def forward(self, graph, in_feat):
        # fix for arxiv dataset
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        
        start = time.time()
        layerTimes = []
        for i, layer in enumerate(self.layers):
            startLayer = time.time()
            h = layer(graph, in_feat if i == 0 else h)
            # h = F.relu(h) not sure if this is needed
            if i == 1:
                h = h.mean(1)
            else:
                h = h.flatten(1)
            layerTimes.append(time.time() - startLayer)

        return h, layerTimes, time.time() - start

class dgl_SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_size, hid_size[0], "gcn"))
        for i in range(len(hid_size)-1):
            self.layers.append(SAGEConv(hid_size[i], hid_size[i+1], "gcn"))
        self.layers.append(SAGEConv(hid_size[-1], out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, in_feat):
        # fix for arxiv dataset
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        
        start = time.time()
        layerTimes = []
        h = self.dropout(in_feat)
        for l, layer in enumerate(self.layers):
            startLayer = time.time()
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
            layerTimes.append(time.time() - startLayer)

        return h, layerTimes, time.time() - start
    
class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class dgl_GIN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()

        input_dim = in_size
        hidden_dim = hid_size[0]
        output_dim = out_size
        num_layers = len(hid_size)

        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = len(hid_size)+1
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):

        start = time.time()
        layerTimes = []
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            startLayer = time.time()
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
            layerTimes.append(time.time() - startLayer)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            ## SOMETHING WRONG HERE
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](h))
        return score_over_layer, layerTimes, time.time() - start