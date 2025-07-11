import argparse
import timeit
import numpy as np
import dgl
import dgl.data
from dgl.data import DGLDataset
import pandas as pd
import torch
import gc
import time

import torch as th
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair

import dgl.nn as dglnn


import dgl
from dgl import function as fn
from dgl.utils import expand_as_pair

def load_ogb(name, root="../../Data/ogb/"):
    from ogb.nodeproppred import DglNodePropPredDataset

    # print("load", name)
    data = DglNodePropPredDataset(name=name, root=root)
    # print("finish loading", name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata["label"] = labels
    in_feats = graph.ndata["feat"].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    train_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    # print("finish constructing", name)
    return graph, num_labels

def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

class GCNOpt(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_hidden,
                 h_layers,
                 graph,
                 feat,
                 log_e2e_time=False,
                 discard_k=2,
                 device_str="cpu",
                 use_opt=False):
        super().__init__()
        self.discard_k = discard_k
        self.layers = nn.ModuleList()
        self.log_e2e_time = log_e2e_time
        self.device_str = device_str

        self.layers = nn.ModuleList()
        if h_layers == 0:
            self.layers.append(dglnn.GraphConv(in_feats=in_feats, out_feats=out_feats, bias=False, activation=F.relu, allow_zero_in_degree=True))
        else:
            self.layers.append(dglnn.GraphConv(in_feats=in_feats, out_feats=n_hidden, bias=False, activation=F.relu, allow_zero_in_degree=True))
            for layer in range(h_layers - 1):
                self.layers.append(dglnn.GraphConv(in_feats=n_hidden, out_feats=n_hidden, bias=False, activation=F.relu, allow_zero_in_degree=True))
            self.layers.append(dglnn.GraphConv(in_feats=n_hidden, bias=False, out_feats=out_feats, allow_zero_in_degree=True))

        if log_e2e_time:
            # time taken for forward gat
            self.needs_lazy_update = True
            self._time_stats = {"fgat": 0.0, "call_count": 0, "iter": []}

    def forward(self, graph, features):
        if self.log_e2e_time:
            if not self.needs_lazy_update:
                n = self._time_stats["call_count"]
                if n > (self.discard_k+1):
                    self._time_stats["fgat"] = self._time_stats["fgat"] * (n-self.discard_k)
                self.needs_lazy_update = True

        # Discard upto kth observation
        if self.log_e2e_time:
            if self._time_stats["call_count"] <= self.discard_k:
                self.reset_timers()

        h = features

        if self.log_e2e_time:
            if self.device_str == "cuda":
                torch.cuda.synchronize()
            t_start = timeit.default_timer()

        for layer in self.layers:
            h = layer(graph, h)

        if self.log_e2e_time:
            if self.device_str == "cuda":
                torch.cuda.synchronize()
            t_end = timeit.default_timer()
            self.update_timer("fgat", t_end-t_start)
            self._time_stats["iter"].append(t_end-t_start)

        if self.log_e2e_time:
            self.update_timer("call_count")

        return h

    def reset_timers(self):
        self._time_stats["fgat"] = 0.0
        self._time_stats["iter"] = []

    def update_timer(self, key: str, tn=0.0):

        if key == "fgat":
            # Maintain a running sum
            an = self._time_stats[key]
            n = self._time_stats["call_count"]

            if n >= 0:
                self._time_stats[key] = (an + tn)
            else:
                raise ValueError('Invalid value for call count')

        elif key == "call_count":
            self._time_stats[key] = self._time_stats[key] + 1

    def get_time_stats(self,):
        if self.needs_lazy_update:
            n = self._time_stats["call_count"]

            # Only need to divide if call count is greater than k
            if n > (self.discard_k+1):
                self._time_stats["fgat"] = self._time_stats["fgat"] / (n-self.discard_k)
            self.needs_lazy_update = False
        return self._time_stats


def export_dense_mm(data, filesuffix, path):
    with open(path + filesuffix, 'wb') as file:
        np.save(file, data)


# Graph has already been sent to the device - so just continue
def full_trainer(graph,
                 device_str,
                 n_hidden,
                 n_classes,
                 h_layers,
                 n_epochs,
                 dataset,
                 log_e2e_time=True,
                 train_model=False,
                 discard_k=2,
                 use_opt=False):
    results = {}

    features = graph.ndata["feat"]
    labels = graph.ndata["label"]

    if (str(args.dataset) == "CoraFullDataset"):
        n_nodes = features.shape[0]
        train_ratio = 0.7
        val_ratio = 0.15
        # test_ratio = 0.15

        # Create indices for the dataset
        indices = torch.randperm(n_nodes)

        # Calculate the sizes of each split
        train_size = int(train_ratio * n_nodes)
        val_size = int(val_ratio * n_nodes)
        # test_size = n_nodes - train_size - val_size

        # Split the indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create masks for each split
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        # Assign True to the corresponding indices for each split
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
    else:
        train_mask = graph.ndata["train_mask"]
        val_mask = graph.ndata["val_mask"]
        test_mask = graph.ndata["test_mask"]

    with graph.local_scope():
        device = torch.device(device_str)
        t0 = time.time()
        model = GCNOpt(in_feats=features.shape[1],
                       out_feats=n_classes,
                       n_hidden=n_hidden,
                       h_layers=h_layers,
                       graph=graph,
                       feat=features,
                       log_e2e_time=log_e2e_time,
                       device_str=device_str,
                       use_opt=use_opt)
        t1 = time.time()
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.01, weight_decay=5e-4)

        features = graph.ndata["feat"]

        epoch_times = []
        epoch_memory = []

        max_acc = 0
        max_acc_epoch = 0

        for epoch in range(n_epochs):
            if device_str == "cuda":
                torch.cuda.synchronize()
            t_start = timeit.default_timer()

            if train_model:
                model.train()
            else:
                model.eval()

            logits = model(graph, features)

            if train_model:
                loss = criterion(logits[train_mask], labels[train_mask])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if device_str == "cuda":
                torch.cuda.synchronize()
            t_end = timeit.default_timer()

            if epoch > discard_k:
                epoch_times.append(t_end - t_start)
                epoch_memory.append(torch.cuda.max_memory_allocated(device=None)/(1024*1024))

            if train_model:
                acc = evaluate(graph, features, labels, test_mask, model)
                # if acc > max_acc:
                max_acc = acc
                max_acc_epoch = epoch

        running_results = model.get_time_stats()
        results["call_count"] = running_results["call_count"]
        results["fgat"] = running_results["fgat"]
        results["iter"] = running_results["iter"]
        results["time_mean"] = np.mean(epoch_times)
        results["memory_mean"] = np.mean(epoch_memory)
        results["time_init"] = t1 - t0
        results["acc"] = max_acc
        results["acc_epoch"] = max_acc_epoch
    return results


def main(args):
    # If dataset does not exist an error is raised.
    initial_features_for_suite = args.n_hidden
    initial_classes_for_suite = args.n_hidden
    dataset_name = getattr(dgl.data, str(args.dataset), False)
    if not dataset_name:
        ogbn_data = ['ogbn-proteins', 'ogbn-products', 'ogbn-arxiv', 'ogbn-mag', 'ogbn-papers100M']
        if (args.dataset in ogbn_data):
            graph, n_classes = load_ogb(args.dataset)
    else:
        dataset = dataset_name()
        graph = dataset[0]
        n_classes = dataset.num_classes

    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    # graph = dgl.reverse(graph, copy_edata=True)

    # Set device
    device_str = args.device
    if device_str == "cuda" and (not torch.cuda.is_available()):
        print("GPU is not available for benchmarking - switching to CPU")
        device_str = "cpu"

    # Read and define params
    n_rows = graph.num_nodes()
    n_hidden = args.n_hidden
    n_epochs = args.n_epochs

    n_edges = graph.number_of_edges()

    ##########################################################################################
    device = torch.device(device_str)

    # Send graph to GPU
    graph = graph.to(device)

    train_model = True
    if args.skip_train:
        train_model = False


    r1 = full_trainer(graph=graph,
                      device_str=device_str,
                      n_hidden=n_hidden,
                      n_classes=n_classes,
                      h_layers=args.layers,
                      n_epochs=n_epochs,
                      dataset=str(args.dataset),
                      log_e2e_time=True,
                      train_model=train_model,
                      discard_k=args.discard,
                      use_opt=False)
    log_file_ptr = open(args.logfile, 'a+')
    log_file_ptr.write(str(r1['memory_mean']) + "," + str(r1['time_mean']) + "\n")
    log_file_ptr.close()

    # print(str(r1['time_mean']),",",str(np.mean(r1['iter'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="CoraGraphDataset",
                        help="Dataset name")
    parser.add_argument("--logfile", type=str, default='logger.txt',
                        help="Logging file")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=20,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=32,
                        help="number of input gcn units")
    parser.add_argument("--layers", type=int, default=1,
                        help="number of output gcn units")
    parser.add_argument("--discard", type=int, default=2,
                        help="number of results to discard considering warmup")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--no-self-loop", action='store_true',
                        help="remove graph self-loop (default=False)")
    parser.set_defaults(no_self_loop=False)
    parser.add_argument("--skip_train", action='store_true',
                        help="only store pre-train data (default=False)")
    parser.set_defaults(skip_train=False)
    parser.add_argument("--path", type=str, default="../matrix_data/",
                        help="Destination path for output mtx.")
    parser.add_argument("--export", action='store_true',
                        help="export dgl data to .mtx files (default=False)")
    parser.set_defaults(export=False)
    parser.add_argument("--no_adj", action='store_true',
                        help="when exporting dgl data skip the adj matrix(default=False)")
    parser.set_defaults(no_adj=False)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Select device to perform computations on")
    args = parser.parse_args()
    main(args)