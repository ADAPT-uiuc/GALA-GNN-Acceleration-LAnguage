from numpy import product
import torch
import dgl
import os
from ogb.nodeproppred import DglNodePropPredDataset


def get_dataset(dataset_name: str):
    """Return DGLGraph object from dataset name.

    Parameters
    ----------
    dataset_name: str
        The dataset name.

    Returns
    -------
    DGLGraph
        The graph object.
    """
    home = os.path.expanduser("~")
    ogb_path = os.path.join(home, "ogb")
    ogb_path = "../../Data/ogb/"
    if not os.path.exists(ogb_path):
        os.makedirs(ogb_path)
    if dataset_name == "pubmed":
        pubmed = dgl.data.PubmedGraphDataset()
        g = pubmed[0].int()
        split_idx = {
            "train": g.ndata["train_mask"],
            "valid": g.ndata["val_mask"],
            "test": g.ndata["test_mask"],
        }
        return g, g.ndata["feat"], g.ndata["label"], split_idx, pubmed.num_labels
    elif dataset_name == "cora":
        cora = dgl.data.CoraGraphDataset()
        g = cora[0].int()
        split_idx = {
            "train": g.ndata["train_mask"],
            "valid": g.ndata["val_mask"],
            "test": g.ndata["test_mask"],
        }
        return g, g.ndata["feat"], g.ndata["label"], split_idx, cora.num_labels
    elif dataset_name == "corafull":
        cora = dgl.data.CoraFullDataset()
        g = cora[0].int()
        n_nodes = g.num_nodes()
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
        split_idx = {
            "train": train_mask,
            "valid": val_mask,
            "test": test_mask,
        }
        return g, g.ndata["feat"], g.ndata["label"], split_idx, cora.num_classes
    elif dataset_name == "citeseer":
        citeseer = dgl.data.CiteseerGraphDataset()
        g = citeseer[0].int()
        split_idx = {
            "train": g.ndata["train_mask"],
            "valid": g.ndata["val_mask"],
            "test": g.ndata["test_mask"],
        }
        return g, g.ndata["feat"], g.ndata["label"], split_idx, citeseer.num_labels
    elif dataset_name == "arxiv":
        arxiv = DglNodePropPredDataset(name="ogbn-arxiv", root=ogb_path)
        g = arxiv[0][0].int()
        return (
            g,
            g.ndata["feat"],
            arxiv.labels.squeeze(-1),
            arxiv.get_idx_split(),
            arxiv.num_classes,
        )
    elif dataset_name == "proteins":
        proteins = DglNodePropPredDataset(name="ogbn-proteins", root=ogb_path)
        g = proteins[0][0].int()
        return (
            g,
            g.ndata["feat"],
            proteins.labels.squeeze(-1),
            proteins.get_idx_split(),
            proteins.num_classes,
        )
    elif dataset_name == "products":
        products = DglNodePropPredDataset(name="ogbn-products", root=ogb_path)
        g = products[0][0].int()
        return (
            g,
            g.ndata["feat"],
            products.labels.squeeze(-1),
            products.get_idx_split(),
            products.num_classes,
        )
    elif dataset_name == "ppi":
        ppi = dgl.data.PPIDataset()
        g = dgl.batch(ppi).int()
        split_idx = {
            "train": torch.ones(g.num_nodes()).bool(),
            "valid": torch.zeros(g.num_nodes()).bool(),
            "test": torch.zeros(g.num_nodes()).bool(),
        }
        return g, g.ndata["feat"], g.ndata["label"], split_idx, ppi.num_labels
    elif dataset_name == "reddit":
        reddit = dgl.data.RedditDataset()
        g = reddit[0].int()
        split_idx = {
            "train": g.ndata["train_mask"],
            "valid": g.ndata["val_mask"],
            "test": g.ndata["test_mask"],
        }
        return g, g.ndata["feat"], g.ndata["label"], split_idx, reddit.num_classes
    else:
        raise KeyError("Unknown dataset: {}".format(dataset_name))