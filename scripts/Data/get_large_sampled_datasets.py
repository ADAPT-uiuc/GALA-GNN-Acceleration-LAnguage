import argparse
import timeit
import numpy as np
import dgl
import dgl.data
from dgl.data import DGLDataset
import torch
import gc
import time

import torch as th

import dgl
import os

def load_ogb(name, root="/shared/damitha2/ogb"):
    from ogb.nodeproppred import DglNodePropPredDataset

    print("load", name)
    data = DglNodePropPredDataset(name=name, root=root)
    print("finish loading", name)
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
    print("finish constructing", name)
    return graph, num_labels


def export_dense_mm(data, filesuffix, path):
    with open(path + filesuffix, 'wb') as file:
        np.save(file, data)

def main(args):
    dataset_name = getattr(dgl.data, str(args.dataset), False)
    print(str(args.dataset))
    if not dataset_name:
        ogbn_data = ['ogbn-proteins', 'ogbn-products', 'ogbn-arxiv', 'ogbn-mag', 'ogbn-papers100M']
        if (args.dataset in ogbn_data):
            src_graph, n_classes = load_ogb(args.dataset)
    else:
        dataset = dataset_name()
        src_graph = dataset[0]
        n_classes = dataset.num_classes

    src_graph = dgl.remove_self_loop(src_graph)
    src_graph = dgl.add_self_loop(src_graph)

    percen = [1, 2, 5, 10, 20]
    nrows = src_graph.num_nodes()

    # Saving the Graph's masks.
    if (str(args.dataset) == "CoraFullDataset"):
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
        train_mask = src_graph.ndata['train_mask']
        val_mask = src_graph.ndata['val_mask']
        test_mask = src_graph.ndata['test_mask']

    for pi in percen:
        true_indices_train = torch.nonzero(train_mask, as_tuple=True)[0]
        true_indices_val = torch.nonzero(val_mask, as_tuple=True)[0]
        true_indices_test = torch.nonzero(test_mask, as_tuple=True)[0]
        others_mask = train_mask | val_mask | test_mask
        others_mask = ~others_mask
        true_indices_others = torch.nonzero(others_mask, as_tuple=True)[0]

        num_to_flip_train = int(len(true_indices_train) * pi // 100)
        num_to_flip_val = int(len(true_indices_val) * pi // 100)
        num_to_flip_test = int(len(true_indices_test) * pi // 100)
        num_to_flip_others = int(len(true_indices_others) * pi // 100)

        flip_indices_train = true_indices_train[:num_to_flip_train]
        flip_indices_val = true_indices_val[:num_to_flip_val]
        flip_indices_test = true_indices_test[:num_to_flip_test]

        train_mask_p = torch.zeros(nrows, dtype=torch.bool)
        train_mask_p[flip_indices_train] = True
        val_mask_p = torch.zeros(nrows, dtype=torch.bool)
        val_mask_p[flip_indices_val] = True
        test_mask_p = torch.zeros(nrows, dtype=torch.bool)
        test_mask_p[flip_indices_test] = True

        # if (num_to_flip_others > 0):
        # flip_indices_others = true_indices_others[:num_to_flip_others]
        flip_indices_others = true_indices_others[-num_to_flip_others:]
        others_mask_p = torch.zeros(nrows, dtype=torch.bool)
        others_mask_p[flip_indices_others] = True
        nodes = train_mask_p | val_mask_p | test_mask_p | others_mask_p
        # else:
        #     nodes = train_mask_p | val_mask_p | test_mask_p

        graph = dgl.node_subgraph(src_graph, nodes)

        n_edges = graph.number_of_edges()

        print(args.dataset, n_classes, graph.num_nodes(), n_edges, graph.ndata['feat'].size(), graph.ndata['label'].size(), num_to_flip_train, num_to_flip_val, num_to_flip_test, num_to_flip_others)

        output_path = r"/shared/damitha2/gala_npy/" + args.dataset + "_" + str(pi)

        print("Exporting", args.dataset, "to", output_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        n_nodes = graph.num_nodes()
        n_edges = graph.number_of_edges()

        # Export adjacency matrix
        src_node, dst_node = graph.edges(form='uv', order='srcdst')
        src_node = np.append([graph.num_nodes(), graph.num_nodes()], src_node.numpy())
        dst_node = dst_node.numpy()
        with open(output_path + "/Adj_src.npy", 'wb') as file_src:
            np.save(file_src, src_node.astype(np.uint32))
        with open(output_path + "/Adj_dst.npy", 'wb') as file_dst:
            np.save(file_dst, dst_node.astype(np.uint32))
        print("complete Adj src, dst")

        emb_data = graph.ndata['feat'].numpy().astype(np.float32)
        export_dense_mm(emb_data, "/Feat.npy", output_path)
        print("complete feats")

        # Saving the Graph's labels (dense vectors but saved as dense matrix).
        label_data = graph.ndata['label'].numpy().astype(np.int64)
        label_data = label_data.reshape((label_data.shape[0], 1))
        export_dense_mm(label_data, "/Lab.npy", output_path)
        print("complete labels")

        train_mask_data = graph.ndata['train_mask'].numpy().astype(np.int32)
        train_mask_data = train_mask_data.reshape((train_mask_data.shape[0], 1))
        export_dense_mm(train_mask_data, "/TnMsk.npy", output_path)
        val_mask_data = graph.ndata['val_mask'].numpy().astype(np.int32)
        val_mask_data = val_mask_data.reshape((val_mask_data.shape[0], 1))
        export_dense_mm(val_mask_data, "/VlMsk.npy", output_path)
        test_mask_data = graph.ndata['test_mask'].numpy().astype(np.int32)
        test_mask_data = test_mask_data.reshape((test_mask_data.shape[0], 1))
        export_dense_mm(test_mask_data, "/TsMsk.npy", output_path)
        print("complete masks")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="RedditDataset",
                        help="Dataset name")
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