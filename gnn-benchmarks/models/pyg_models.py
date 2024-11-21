import torch as th
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential
from torch.nn import BatchNorm1d as BatchNorm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_add_pool
import time

'''
layerTimes has every forward layer time in array, size is number of layers
forwardTime is a float representing total foward propogation time
'''
class pyg_GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_size, hid_size[0]))
        for i in range(len(hid_size)-1):
            self.layers.append(GCNConv(hid_size[i], hid_size[i+1]))
        self.layers.append(GCNConv(hid_size[-1], out_size))

    def forward(self, graph, in_feat=None):
        start = time.time()
        layerTimes = []
        if in_feat == None:
            in_feat = graph.x
        edge_index = graph.edge_index

        for i, layer in enumerate(self.layers):
            startLayer = time.time()
            x = layer(in_feat if i == 0 else x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            layerTimes.append(time.time() - startLayer)

        return F.log_softmax(x, dim=1), layerTimes, time.time() - start
    
class pyg_GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_size, hid_size[0], heads[0], dropout=0.6))
        for i in range(len(hid_size)-1):
            self.layers.append(GATConv(hid_size[i]*heads[i], hid_size[i+1], heads[i+1], dropout=0.6))
        self.layers.append(GATConv(hid_size[-1]*heads[-2], out_size, heads[-1], dropout=0.6))

    def forward(self, graph, in_feat=None):
        start = time.time()
        layerTimes = []
        if in_feat == None:
            in_feat = graph.x
        edge_index = graph.edge_index
        for i, layer in enumerate(self.layers):
            startLayer = time.time()
            x = F.dropout(in_feat if i == 0 else x, p=0.6, training=self.training)
            x = layer(x, edge_index)
            x = F.elu(x) if i < len(self.layers)-1 else x
            layerTimes.append(time.time() - startLayer)

        return x, layerTimes, time.time() - start

class pyg_SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_size, hid_size[0]))
        for i in range(len(hid_size)-1):
            self.layers.append(SAGEConv(hid_size[i], hid_size[i+1]))
        self.layers.append(SAGEConv(hid_size[-1], out_size))
        
    def forward(self, graph, in_feat=None):
        start = time.time()
        layerTimes = []
        if in_feat == None:
            in_feat = graph.x
        edge_index = graph.edge_index
        for i, layer in enumerate(self.layers):
            startLayer = time.time()
            x = F.dropout(in_feat if i == 0 else x, p=0.6, training=self.training)
            x = layer(x, edge_index)
            x = F.elu(x) if i < len(self.layers)-1 else x
            layerTimes.append(time.time() - startLayer)
        
        return F.log_softmax(x), layerTimes, time.time() - start
    
class pyg_GIN(th.nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()

        in_channels = in_size
        hidden_channels = hid_size[0]
        out_channels = out_size
        num_layers = len(hid_size)

        self.convs = th.nn.ModuleList()
        self.batch_norms = th.nn.ModuleList()

        for i in range(num_layers):
            mlp = Sequential(
                Linear(in_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True)

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))

            in_channels = hidden_channels

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, graph, in_feat=None): # batch was a parameter, removed
        start = time.time()
        layerTimes = []
        x = graph.x
        edge_index = graph.edge_index
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            startLayer = time.time()
            x = F.relu(batch_norm(conv(x, edge_index)))
            layerTimes.append(time.time() - startLayer)
        #x = global_add_pool(x, batch)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), layerTimes, time.time() - start