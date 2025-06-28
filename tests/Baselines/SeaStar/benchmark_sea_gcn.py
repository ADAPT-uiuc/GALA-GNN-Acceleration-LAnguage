import argparse
import timeit
import numpy as np
import dgl.data
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import time

import sys
import os

from dgl.data import load_data
# from ..graph_utils.gala_graphconv_sea import TimedGraphConv
from gala_graphconv_sea import TimedGraphConv as TimedGraphConv


from ogb.nodeproppred import DglNodePropPredDataset

from dgl import transform
from dgl import DGLGraph
import dgl
from seastar import CtxManager

def evaluate(g, features, labels, mask, model, norm_data):
    model.eval()
    with torch.no_grad():
        logits = model(g, features, norm_data)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


class EdgeGCNetwork(torch.nn.Module):
    def __init__(self,
                in_feats,
                out_feats,
                n_hidden,
                h_layers,
                log_e2e_time=False,
                use_sddmm=False,
                use_default=True,
                discard_k=2,
                device_str="cpu"):
        super().__init__()
        self.discard_k = discard_k
        self.layers = nn.ModuleList()
        self.log_e2e_time = log_e2e_time
        self.use_sddmm = use_sddmm
        self.device_str = device_str

        self.layers = nn.ModuleList()
        if h_layers == 0:
            self.layers.append(TimedGraphConv(in_feats=in_feats, out_feats=out_feats, activation=F.relu, allow_zero_in_degree=True))
        else:
            self.layers.append(TimedGraphConv(in_feats=in_feats, out_feats=n_hidden, activation=F.relu, allow_zero_in_degree=True))
            for layer in range(h_layers - 1):
                self.layers.append(TimedGraphConv(in_feats=n_hidden, out_feats=n_hidden, activation=F.relu, allow_zero_in_degree=True))
            self.layers.append(TimedGraphConv(in_feats=n_hidden, out_feats=out_feats, allow_zero_in_degree=True))

        if log_e2e_time:
            # time taken for forward gat
            self.needs_lazy_update = True
            self._time_stats = {"fgat": 0.0, "call_count": 0, "iter": []}

    def forward(self, graph, feat, norm_data):
        if self.log_e2e_time:
            if not self.needs_lazy_update:
                n = self._time_stats["call_count"]
                if n > (self.discard_k + 1):
                    self._time_stats["fgat"] = self._time_stats["fgat"] * (n - self.discard_k)
                self.needs_lazy_update = True

        # Discard upto kth observation
        if self.log_e2e_time:
            if self._time_stats["call_count"] <= self.discard_k:
                self.reset_timers()

        h = feat

        if self.log_e2e_time:
            if self.device_str == "cuda":
                torch.cuda.synchronize()
            t_start = timeit.default_timer()

        dgl_context = dgl.utils.to_dgl_context(feat.device)
        graph = graph._graph.get_immutable_gidx(dgl_context)

        for layer in self.layers:
            h = layer(graph, h, norm_data)

        if self.log_e2e_time:
            if self.device_str == "cuda":
                torch.cuda.synchronize()
            t_end = timeit.default_timer()
            self.update_timer("fgat", t_end - t_start)
            self._time_stats["iter"].append(t_end - t_start)

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

    def get_time_stats(self, ):
        if not self.log_e2e_time:
            print('Logging time is disabled.')

        if self.needs_lazy_update:
            n = self._time_stats["call_count"]

            # Only need to divide if call count is greater than k
            if n > (self.discard_k + 1):
                self._time_stats["fgat"] = self._time_stats["fgat"] / (n - self.discard_k)
            self.needs_lazy_update = False
        return self._time_stats


# Graph has already been sent to the device - so just continue
def full_trainer(graph,
                 device_str,
                 n_hidden,
                 n_epochs,
                 h_layers,
                 in_dim,
                 out_dim,
                 log_e2e_time=True,
                 use_sddmm=False,
                 use_default=True,
                 train_model=False,
                 discard_k=2):

    results = {}

    best_val_acc = 0
    best_test_acc = 0

    features = graph.ndata["feat"]
    labels = graph.ndata["label"]
    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]

    device = torch.device(device_str)
    model = EdgeGCNetwork(in_dim, out_dim, n_hidden, h_layers, log_e2e_time=log_e2e_time,
                          use_sddmm=use_sddmm, device_str=device_str, use_default=use_default)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)

    # Precomputation step for sddmm - called at the beginning of all computations
    # if use_sddmm:
    # feat_src, feat_dst = expand_as_pair(features, graph)

    sddmm_epoch_times = []
    for epoch in range(n_epochs):
        # Time pre-compute
        deg = graph.in_degrees().cuda()
        if log_e2e_time:
            if device_str == "cuda":
                torch.cuda.synchronize()
            t_start = timeit.default_timer()

        degs = deg.float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm_data = norm.unsqueeze(1)

        if log_e2e_time:
            if device_str == "cuda":
                torch.cuda.synchronize()
            t_end = timeit.default_timer()
            if epoch > discard_k:
                sddmm_epoch_times.append(t_end - t_start)

    results["pre_compute_mean"] = np.mean(sddmm_epoch_times)
    results["pre_compute_std"] = np.std(sddmm_epoch_times)

    epoch_times = []
    model.eval()

    for epoch in range(n_epochs):
        torch.cuda.synchronize()
        t_start = timeit.default_timer()

        if train_model:
            model.train()
        else:
            model.eval()

        logits = model(graph, features, norm_data)

        if train_model:
            # print(logits[train_mask].shape, labels[train_mask].shape, file=sys.stderr)
            loss = criterion(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        t_end = timeit.default_timer()

        if epoch > discard_k:
            epoch_times.append(t_end - t_start)

        # acc = evaluate(graph, features, labels, test_mask, model, norm_data)
        # print(epoch, ":", acc)
        

    running_results = model.get_time_stats()
    results["call_count"] = running_results["call_count"]
    results["fgat"] = running_results["fgat"]
    results["iter"] = running_results["iter"]
    results["time_mean"] = np.mean(epoch_times)
    results["time_std"] = np.std(epoch_times)
    return results


def main(args):
    initial_features_for_suite = args.n_hidden
    initial_classes_for_suite = args.n_hidden
    print(str(args.dataset))

    raw_dir = "../Data/gala_npy/" + args.dataset + "/"
    adj_src_path = raw_dir + "Adj_src.npy"
    adj_dst_path = raw_dir + "Adj_dst.npy"
    feat_path = raw_dir + "Feat.npy"
    lab_path = raw_dir + "Lab.npy"

    tnMask_path = raw_dir + "TnMsk.npy"
    vlMask_path = raw_dir + "VlMsk.npy"
    tsMask_path = raw_dir + "TsMsk.npy"

    graph = DGLGraph()

    adj_src_data = np.load(adj_src_path)
    num_nodes = adj_src_data[0]
    adj_src_data = adj_src_data[2:]
    adj_dst_data = np.load(adj_dst_path)
    feat_data = np.load(feat_path)
    lab_data = np.load(lab_path)
    lab_data = lab_data.squeeze(1)

    print(feat_data.shape, lab_data.shape)
    in_size = feat_data.shape[1]
    n_classes = max(lab_data) + 1
    # n_classes = n_classes[0]
    print("cls", n_classes)

    n_rows = num_nodes
    n_hidden = args.n_hidden
    n_epochs = args.n_epochs
    n_layers = args.n_layers

    # in_size = 32
    # n_classes = 32

    #     # Define features and classes
    # feat_data = torch.tensor(np.random.rand(n_rows, in_size), dtype=torch.float)
    # lab_data = torch.tensor(np.random.randint(0, n_classes, size=n_rows))

    graph.add_nodes(num_nodes)
    graph.add_edges(adj_src_data, adj_dst_data)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Set device
    device_str = args.device
    if device_str == "cuda" and (not torch.cuda.is_available()):
        print("GPU is not available for benchmarking - switching to CPU")
        device_str = "cpu"

    print("Selected device", device_str)

    graph.ndata["feat"] = feat_data
    graph.ndata["label"] = lab_data

    # train_ratio = 0.6
    # val_ratio = 0.2
    # ## Masks indicating whether a node belongs to training, validation, or test set
    # n_train = int(num_nodes * train_ratio)
    # n_val = int(num_nodes * val_ratio)
    # train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # train_mask[:n_train] = True
    # val_mask[n_train: n_train + n_val] = True
    # test_mask[n_train + n_val:] = True
    # graph.ndata["train_mask"] = train_mask
    # graph.ndata["val_mask"] = val_mask
    # graph.ndata["test_mask"] = test_mask

    # print("AA", graph.ndata["train_mask"].shape)

    graph.ndata["train_mask"] = np.load(tnMask_path).squeeze(1)
    graph.ndata["val_mask"] = np.load(vlMask_path).squeeze(1)
    graph.ndata["test_mask"] = np.load(tsMask_path).squeeze(1)

    # print("AB", graph.ndata["train_mask"].shape)

    graph.num_classes = n_classes
    print("Initialized features")

    n_edges = graph.number_of_edges()

    print("""----Data statistics------'
    #Edges %d
    #Classes %d""" %
          (n_edges, n_classes,
           ))

    ##########################################################################################
    device = torch.device(device_str)

    # Send graph to GPU
    graph = graph.to(device)

    train_model = True
    if args.skip_train:
        train_model = False

    # Edge-SPMM
    print('================Timing edge-SPMM==============')
    r1 = full_trainer(graph=graph,
                      device_str=device_str,
                      n_hidden=n_hidden,
                      n_epochs=n_epochs,
                      h_layers=n_layers,
                      in_dim=in_size,
                      out_dim=n_classes,
                      log_e2e_time=True,
                      use_sddmm=False,
                      use_default=True,
                      train_model=train_model,
                      discard_k=args.discard)

    log_file_ptr = open(args.logfile, 'a+')
    log_file_ptr.write(str(np.mean(r1['iter'])) + "," + str(r1['time_mean']) + "\n")
    log_file_ptr.close()


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
    parser.add_argument("--n-hidden", type=int, default=256,
                        help="number of input gcn units")
    parser.add_argument("--discard", type=int, default=2,
                        help="number of results to discard considering warmup")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
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
