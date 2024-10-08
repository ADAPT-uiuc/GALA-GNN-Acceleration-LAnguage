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
            graph, n_classes = load_ogb(args.dataset)
    else:
        dataset = dataset_name()
        graph = dataset[0]
        n_classes = dataset.num_classes

    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)

    output_path = r"/shared/damitha2/gala_npy/" + args.dataset

    print("Exporting", args.dataset, "to", output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

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

    # Saving the Graph's masks.
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