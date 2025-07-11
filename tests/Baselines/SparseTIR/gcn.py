import argparse
import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.dlpack import to_dlpack as th_to_dlpack
from torch.utils.dlpack import from_dlpack as th_from_dlpack

import time
from utils import get_dataset
import tvm
import tvm.testing
import tvm.tir as tir
from tvm.script import tir as T
from tvm.sparse import (
    FormatRewriteRule,
    lower_sparse_buffer,
    lower_sparse_iter,
    column_part_hyb,
    format_decompose,
)


@T.prim_func
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    num_tiles: T.int32,
    nnz: T.int32,
    cwm: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    K1 = T.dense_fixed(num_tiles)
    K2 = T.dense_fixed(cwm)
    K3 = T.dense_fixed(32)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K1, K2, K3), "float32")
    C = T.match_sparse_buffer(c, (I, K1, K2, K3), "float32")
    with T.iter([I, J, K1, K2, K3], "SRSSS", "csrmm") as [i, j, k1, k2, k3]:
        with T.init():
            C[i, k1, k2, k3] = 0.0
        C[i, k1, k2, k3] = C[i, k1, k2, k3] + A[i, j] * B[j, k1, k2, k3]


@T.prim_func
def ell(
    a: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indices_j: T.handle,
    m: T.int32,
    n: T.int32,
    num_rows: T.int32,
    nnz_cols: T.int32,
) -> None:
    O = T.dense_fixed(1)
    I = T.sparse_variable(O, (m, num_rows), (indptr_i, indices_i))
    J = T.sparse_fixed(I, (n, nnz_cols), indices_j)
    A = T.match_sparse_buffer(a, (O, I, J), "float32")
    T.evaluate(0)


def csr2ell_inv_index_map(o, i, j):
    return i, j


def csr2ell_index_map(i, j):
    return 0, i, j


kernels = {}
kernel_args = {}
forward_pass_times = []

class SpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        X_nd = tvm.nd.from_dlpack(th_to_dlpack(X.view(-1).contiguous()))
        Y = torch.zeros_like(X)
        Y_nd = tvm.nd.from_dlpack(th_to_dlpack(Y.view(-1).contiguous()))
        f = kernels[(X.shape[-1], True)]
        args = [X_nd, Y_nd]
        args += kernel_args[True]
        f(*args)
        return Y

    @staticmethod
    def backward(ctx, dY):
        dY_nd = tvm.nd.from_dlpack(th_to_dlpack(dY.view(-1).contiguous()))
        dX = torch.zeros_like(dY)
        dX_nd = tvm.nd.from_dlpack(th_to_dlpack(dX.view(-1).contiguous()))
        # The graph we profiled are undirected.
        # f = kernels[(dY.shape[-1], False)]
        f = kernels[(dY.shape[-1], True)]
        args = [dY_nd, dX_nd]
        # args += kernel_args[False]
        args += kernel_args[True]
        f(*args)
        return dX


class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, norm="both", weight=True, bias=True, activation=None):
        super(GraphConv, self).__init__()

        self._in_feats = in_feats
        self._out_feats = out_feats
        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self._norm = norm
        self._weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        #self._bias = nn.Parameter(torch.Tensor(out_feats))
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, feat, graph):
        with graph.local_scope():
            '''
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )
            '''
            aggregate_fn = fn.copy_u("h", "m")
            '''
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")
            '''

            weight = self._weight
            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                graph.srcdata["h"] = feat_src
                #graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = SpMM.apply(feat_src) # <--- sparsetir integration 
                #rst = graph.dstdata["h"]
                graph.dstdata["h"] = rst
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_src
                #graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = SpMM.apply(feat_src) # <--- sparstir integration
                #rst = graph.dstdata["h"]
                graph.dstdata["h"] = rst
                if weight is not None:
                    rst = torch.matmul(rst, weight)

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            #if self._bias is not None:
                #rst = rst + self._bias

            if self._activation is not None:
                rst = self._activation(rst)
            
            return rst

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers, dropout, g):
        super(GCN, self).__init__()

        self.graph = g
        self.layers = nn.ModuleList()
        #self.bns = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, hidden_feats))
        #self.bns.append(nn.BatchNorm1d(hidden_feats))
        # hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GraphConv(hidden_feats, hidden_feats))
            #self.bns.append(nn.BatchNorm1d(hidden_feats))
        # output layer
        self.layers.append(GraphConv(hidden_feats, out_feats))
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        #for bn in self.bns:
            #bn.reset_parameters()

    def forward(self, x):
        torch.cuda.synchronize()
        start = time.time()
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x,self.graph)
            #print(time.time()-start)
            #x = self.bns[i](x)
            x = F.relu(x)
            #x = self.dropout(x)
        x = self.layers[-1](x,self.graph)
        torch.cuda.synchronize()
        forward_pass_times.append(time.time()-start)

        return x.log_softmax(dim=-1)


def train(dataset, model, feats, y_true, train_idx, optimizer,graph):
    model.train()

    optimizer.zero_grad()
    out = model(feats)[train_idx]
    '''
    if dataset == "ppi":
        loss = F.binary_cross_entropy_with_logits(out, y_true[train_idx])
    else:
        loss = F.nll_loss(out, y_true[train_idx])
    '''
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, y_true[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def create_kernels(g, feat_sizes, bucket_sizes=[], num_col_parts=1):
    global kernels
    global kernel_args
    use_implicit_unroll = True

    for forward in [True]:  # [True, False]
        num_buckets = len(bucket_sizes)
        indptr, indices, _ = g.adj_sparse("csc")
        m = g.num_dst_nodes()
        n = g.num_src_nodes()
        nnz = g.num_edges()
        indptr_nd = tvm.nd.array(indptr.cpu().numpy(), device=tvm.cpu())
        indices_nd = tvm.nd.array(indices.cpu().numpy(), device=tvm.cpu())
        row_indices, col_indices, mask = column_part_hyb(
            m, n, indptr_nd, indices_nd, num_col_parts, bucket_sizes
        )

        # prepare nd array

        args = []
        for part_id in range(num_col_parts):
            for bucket_id, _ in enumerate(bucket_sizes):
                weight = tvm.nd.array(
                    mask[part_id][bucket_id].numpy().reshape(-1).astype("float32"),
                    device=tvm.cuda(0),
                )
                rows = tvm.nd.array(
                    row_indices[part_id][bucket_id].numpy().astype("int32"),
                    device=tvm.cuda(0),
                )
                cols = tvm.nd.array(
                    col_indices[part_id][bucket_id].numpy().reshape(-1).astype("int32"),
                    device=tvm.cuda(0),
                )
                args += [weight, rows, cols]

        kernel_args[forward] = args

        for feat_size in feat_sizes:
            if feat_size <= 32:
                coarsening_factor = 1
            elif feat_size <= 128:
                coarsening_factor = 2
            else:
                coarsening_factor = 4

            # rewrite csrmm
            nnz_cols_symbol = ell.params[-1]
            rewrites = []
            for part_id in range(num_col_parts):
                for bucket_id, bucket_size in enumerate(bucket_sizes):
                    rewrites.append(
                        FormatRewriteRule(
                            str(part_id) + "_" + str(bucket_id),
                            ell.specialize({nnz_cols_symbol: bucket_size}),
                            ["A"],
                            ["I", "J"],
                            ["O", "I", "J"],
                            {"I": ["O", "I"], "J": ["J"]},
                            csr2ell_index_map,
                            csr2ell_inv_index_map,
                        )
                    )
            mod = tvm.IRModule.from_expr(csrmm)
            mod = format_decompose(mod, rewrites)
            mod = tvm.tir.transform.RemovePreprocess()(mod)

            # specialize
            params = mod["main"].params
            param_map = {
                params[5]: m,  # m
                params[6]: n,  # n
                params[7]: feat_size // coarsening_factor // 32,  # num_tiles,
                params[8]: nnz,  # nnz
                params[9]: coarsening_factor,  # coersening_factor
            }
            for part_id in range(num_col_parts):
                for bucket_id in range(num_buckets):
                    param_map[
                        params[10 + 7 * (part_id * num_buckets + bucket_id) + 4]
                    ] = m
                    param_map[
                        params[10 + 7 * (part_id * num_buckets + bucket_id) + 5]
                    ] = n
                    param_map[
                        params[10 + 7 * (part_id * num_buckets + bucket_id) + 6]
                    ] = row_indices[part_id][bucket_id].shape[0]

            mod["main"] = (
                mod["main"].specialize(param_map).with_attr("horizontal_fuse", True)
            )

            # schedule
            sch = tvm.tir.Schedule(mod)
            for sp_iter_name in [
                "csrmm_{}_{}".format(i, j)
                for j in range(num_buckets)
                for i in range(num_col_parts)
            ]:
                sp_iteration = sch.get_sparse_iteration(sp_iter_name)
                o, i, j, k1, k2, k3 = sch.get_sp_iters(sp_iteration)
                sch.sparse_fuse(sp_iteration, [o, i])

            mod = sch.mod
            mod = tvm.sparse.lower_sparse_iter(mod)
            sch = tvm.tir.Schedule(mod)
            for part_id in range(num_col_parts):
                for bucket_id, bucket_size in enumerate(bucket_sizes):
                    is_atomic = num_col_parts > 1 or bucket_id + 1 == num_buckets
                    blk = sch.get_block("csrmm_{}_{}0".format(part_id, bucket_id))
                    i, j, foo, foi, fi = sch.get_loops(blk)
                    sch.reorder(foo, fi, j, foi)
                    if is_atomic:
                        sch.annotate(blk, "atomic", True)
                        write_blk = sch.cache_write(blk, 0, "local")
                        sch.reverse_compute_at(write_blk, fi, True)
                        # sch.unroll(sch.get_loops(write_blk)[-2])
                    sch.bind(fi, "threadIdx.x")
                    sch.bind(foo, "blockIdx.y")
                    sch.unroll(foi)
                    if use_implicit_unroll:
                        sch.annotate(foi, "pragma_unroll_explicit", 0)
                    sch.unroll(j)
                    if use_implicit_unroll:
                        sch.annotate(j, "pragma_unroll_explicit", 0)
                    io, ioi, ii = sch.split(
                        i, [None, bucket_sizes[-1] // bucket_size, 8]
                    )
                    sch.bind(io, "blockIdx.x")
                    sch.bind(ii, "threadIdx.y")
                    init_blk = sch.decompose_reduction(blk, fi)
                    ax0, ax1 = sch.get_loops(init_blk)[-2:]
                    sch.bind(ax0, "threadIdx.x")
                    sch.unroll(ax1)
                    if use_implicit_unroll:
                        sch.annotate(ax1, "pragma_unroll_explicit", 0)

            mod = tvm.sparse.lower_sparse_buffer(sch.mod)
            mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
            f = tvm.build(mod, target="cuda")

            kernels[(feat_size, forward)] = f


def pad_length(x: int):
    if x <= 32:
        return 32
    if x <= 64:
        return 64
    ret = 128
    while ret < x:
        ret = ret + 128
    return ret


bucketing_config = {
    "arxiv": [1, 2, 4, 8, 16, 32],
    "proteins": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "pubmed": [1, 2, 4, 8, 16, 32],
    "ppi": [1, 2, 4, 8, 16, 32],
    "cora": [1, 2, 4],
    "citeseer": [1, 2, 4],
    "reddit": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    "products": [1, 2, 4, 8, 16, 32],
    "corafull": [1, 2, 4, 8, 16, 32, 64],
}

col_part = {
    "arxiv": 1,
    "proteins": 8,
    "pubmed": 1,
    "cora": 1,
    "citeseer": 1,
    "ppi": 8,
    "reddit": 8,
    "products": 16,
    "corafull": 4,
}


def main():
    parser = argparse.ArgumentParser(description="OGBN-Cora (GraphConv Full-Batch)")
    parser.add_argument("-d", "--dataset", type=str, default="reddit")
    parser.add_argument("--logfile", type=str, default='logger.txt',
                        help="Logging file")
    args = parser.parse_args()
    print(args)

    device = torch.device(0)

    g, feats, labels, split_idx, num_classes = get_dataset(args.dataset)
    # pad
    feats_ = torch.zeros([feats.shape[0], pad_length(feats.shape[1])])
    feats_[:, : feats.shape[1]] = feats
    feats = feats_
    if args.dataset == "ppi":
        labels_ = torch.zeros([labels.shape[0], pad_length(num_classes)]).to(labels)
        labels_[:, : labels.shape[1]] = labels
        labels = labels_
    g = dgl.to_bidirected(g)
    g = g.int().to(device)
    feats, labels = feats.to(device), labels.to(device)
    train_idx = split_idx["train"].to(device)
    num_classes = pad_length(num_classes)

    create_kernels(
        g,
        [feats.shape[-1], 32, num_classes],
        bucketing_config[args.dataset],
        col_part[args.dataset],
    )

    model = GCN(
        in_feats=feats.size(-1),
        hidden_feats=32,
        out_feats=num_classes,
        num_layers=2,
        dropout=0,
        g=g,
    ).to(device)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    warmup = 20
    active = 200

    for _ in range(warmup):
        loss = train(args.dataset, model, feats, labels, train_idx, optimizer, g)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(active):
        loss = train(args.dataset, model, feats, labels, train_idx, optimizer, g)
        # model.eval()
        # _ = model(feats)
    end_event.record()
    torch.cuda.synchronize()
    dur = start_event.elapsed_time(end_event) / active
    print("---------------GCN---dataset: {}------SparseTIR---------------".format(args.dataset))
    print("Training time: {} ms/epoch".format(dur))
    print("Forward Pass (training) time: {}".format(np.mean(np.array(forward_pass_times[1:]))*1000))
    print("----------------------------------------------")
    log_file_ptr = open(args.logfile, 'a+')
    log_file_ptr.write(str(np.mean(np.array(forward_pass_times[1:]))*1000) + "," + str(dur) + "\n")
    log_file_ptr.close()



if __name__ == "__main__":
    main()
