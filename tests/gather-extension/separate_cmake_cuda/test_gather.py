import time

import torch
# Don't need because it will be a single program without outputs
import argparse

# TODO the following is not really needed if you're writing to a single file
from gather import GCN
from gather import GCN_DGL

# Need
from dgl.data import CoraGraphDataset, RedditDataset
from dgl.utils import expand_as_pair
from dgl import function as fn
import dgl
import numpy as np

# from ogb.nodeproppred import DglNodePropPredDataset

torch.ops.load_library("build/libgala_cuda.so")


def main(args):
    dataset_name = getattr(dgl.data, str(args.dataset), False)
    # print(str(args.dataset))
    if not dataset_name:
        ogbn_data = ['ogbn-proteins', 'ogbn-products', 'ogbn-arxiv', 'ogbn-mag', 'ogbn-papers100M']
        if (args.dataset in ogbn_data):
            dataset = DglNodePropPredDataset(name=args.dataset, root='/home/damitha2/projects')
            graph, labels = dataset[0]

            graph.ndata["label"] = labels

            n_train = int(graph.num_nodes() * 0.6)
            n_val = int(graph.num_nodes() * 0.2)
            train_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
            val_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
            test_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
            train_mask[:n_train] = True
            val_mask[n_train: n_train + n_val] = True
            test_mask[n_train + n_val:] = True
            graph.ndata["train_mask"] = train_mask
            graph.ndata["val_mask"] = val_mask
            graph.ndata["test_mask"] = test_mask
        else:
            print("Error - Graph not supported")
            return
            # # raw_dir = "../../data_suite/"+args.dataset+"/"
            # raw_dir = "/shared/SENSEi/data_suite/"+args.dataset+"/"
            # data_filename = args.dataset+".mtx"
            # dataset = MTXDataset(name=str(args.dataset), raw_dir=raw_dir, fname=data_filename,
            #                      in_feats=initial_features_for_suite)
            # #in_size = initial_features_for_suite
            # graph = dataset[0]
    else:
        dataset = dataset_name()
        graph = dataset[0]

    # # graph = CoraGraphDataset()
    # graph = RedditDataset()
    # graph = graph[0]

    # # Source nodes for edges (2, 1), (3, 2), (4, 3)
    # src_ids = torch.tensor([0, 1, 2, 3, 4])
    # # Destination nodes for edges (2, 1), (3, 2), (4, 3)
    # dst_ids = torch.tensor([1, 2, 3, 4, 0])
    # graph = dgl.graph((src_ids, dst_ids), num_nodes=5)
    # graph.ndata["feat"] = torch.ones((5, 8))

    nrows = graph.num_nodes()
    nvals = graph.number_of_edges()

    if args.n_input == -1:
        input_dense = graph.ndata["feat"]
        in_feats = input_dense.shape[1]
        input_dense = torch.ones((nrows, in_feats))
    else:
        in_feats = args.n_input
        input_dense = torch.ones((nrows, in_feats))
    labels = graph.ndata["label"]

    feat_src, feat_dst = expand_as_pair(input_dense, graph)
    degs = graph.out_degrees().to(feat_src).clamp(min=1)
    # degs = torch.ones(nrows)
    norm = torch.pow(degs, -0.5)
    graph.srcdata.update({"di": norm})
    graph.dstdata.update({"do": norm})
    graph.apply_edges(fn.u_mul_v("di", "do", "dd"))

    # print(graph.adj_tensors('csr')[0], graph.adj_tensors('csr')[0].shape)
    edges = graph.number_of_edges()

    if args.n_output == -1:
        out_feats = 32
    else:
        out_feats = args.n_output

    iters = args.iters
    skip_iters = args.discard

    # # TODO eval using a DGL graph and have a method to call DGL functions
    # offsets = torch.arange(0, edges, edges_per_node, dtype=torch.int64)
    # cols = torch.randint(0, nodes, (edges,), dtype=torch.int32)
    # vals = torch.randn(edges, dtype=torch.float32)

    offsets = graph.adj_tensors('csr')[0].to(torch.int32)
    cols = graph.adj_tensors('csr')[1].to(torch.int32)
    vals = graph.edata['dd']

    # tile_size = 16000
    # t1_tile_offsets, t1_offsets, t1_rows, t1_cols, t1_vals = torch.ops.gala_ops.tiling_graph(tile_size, offsets, cols, vals)

    criterion = torch.nn.CrossEntropyLoss()
    gnn = GCN(in_feats, out_feats)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01, weight_decay=5e-4)
    forward = 0
    backward = 0
    iter_times_forward = []
    iter_times_backward = []

    input_dense.to('cuda')
    offsets.to('cuda')
    cols.to('cuda')
    vals.to('cuda')

    # torch.set_grad_enabled(True)
    for _iter in range(iters):
        # torch.cuda.synchronize()
        start = time.time()
        new_h_cpp = gnn(input_dense, offsets, cols, vals)
        new_h_cpp = new_h_cpp[0]

        # torch.cuda.synchronize()
        forward = time.time() - start
        if _iter >= skip_iters or iters <= skip_iters:
            iter_times_forward.append(forward)

        start = time.time()
        # (new_h_cpp).backward()

        # loss = criterion(new_h_cpp, input_dense)
        # loss.requires_grad = True
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        backward += time.time() - start
    print("GNN - cuSparse")
    print('Forward: {:.3f} s (std: {:.3f}) | Backward {:.3f} s'.format(np.mean(iter_times_forward), np.std(iter_times_forward), backward))

    gnn_dgl = GCN_DGL(in_feats, out_feats)
    gnn_dgl.eval()
    optimizer_dgl = torch.optim.Adam(gnn_dgl.parameters(), lr=0.01, weight_decay=5e-4)
    forward = 0
    backward = 0
    iter_times_forward = []
    iter_times_backward = []

    for _iter in range(iters):
        torch.cuda.synchronize()
        start = time.time()
        new_h_dgl = gnn_dgl(graph, input_dense)

        torch.cuda.synchronize()
        forward = time.time() - start
        if _iter >= skip_iters or iters <= skip_iters:
            iter_times_forward.append(forward)

        start = time.time()
        # (new_h_dgl.sum()).backward()
        # loss = criterion(new_h_dgl, input_dense)
        # optimizer_dgl.zero_grad()
        # loss.backward()
        # optimizer_dgl.step()
        backward += time.time() - start
    #print('Forward: {:.3f} s | Backward {:.3f} s'.format(forward, backward))
    print("GNN - DGL")
    print('Forward: {:.3f} s (std: {:.3f}) | Backward {:.3f} s'.format(np.mean(iter_times_forward), np.std(iter_times_forward), backward))

    # # ********** Check the validly of the operations ********** #
    # if args.validate_result:
    #     print(new_h_cpp.shape)
    #     # print(new_h_cpp)
    #     print(new_h_cpp_tile.shape)
    #     # print(new_h_cpp_tile)
    #     print(new_h_dgl.shape)
    #     # print(new_h_dgl)
    #
    #     print(torch.isclose(new_h_cpp, new_h_dgl).sum() / (in_feats * nrows))
    #
    #     close_arr = torch.isclose(new_h_cpp, new_h_dgl, rtol=1e-02, atol=1e-04)
    #     close_sum = (close_arr.sum() / (in_feats * nrows))
    #
    #     print("Is close", close_sum)
    #
    #     print("***************")
    #     print(torch.isclose(new_h_cpp_tile, new_h_dgl).sum() / (in_feats * nrows))
    #
    #     close_arr = torch.isclose(new_h_cpp_tile, new_h_dgl, rtol=1e-02, atol=1e-04)
    #     close_sum = (close_arr.sum() / (in_feats * nrows))
    #
    #     print("Is close", close_sum)
    #
    #     # print(close_arr.shape)
    #
    #     count_err = 0
    #     tresh_err = 10
    #
    #     # for i in range(close_arr.shape[0]):
    #     #     count_err = 0
    #     #     for j in range(close_arr.shape[1]):
    #     #         # Accessing each element
    #     #         value = close_arr[i][j].item()  # Accessing element at position (i, j)
    #     #         if (not(value)):
    #     #             print(i, j, new_h_cpp[i][j].item(),  new_h_dgl[i][j].item())
    #     #             count_err += 1
    #     #         if (count_err == tresh_err):
    #     #             break
    #     #     if (count_err == tresh_err):
    #     #         break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="RedditDataset",
                        help="Dataset name")
    parser.add_argument("--n_input", type=int, default=-1,
                        help="input embedding size of the GNN")
    parser.add_argument("--n_output", type=int, default=-1,
                        help="output embedding size of the GNN")
    parser.add_argument("--discard", type=int, default=5,
                        help="number of results to discard considering warmup")
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of iterations to run the GNN")
    parser.add_argument("--validate_result", action='store_true',
                        help="Validate if the result of the implemented methods against DGL")
    parser.set_defaults(no_self_loop=False)
    args = parser.parse_args()
    main(args)