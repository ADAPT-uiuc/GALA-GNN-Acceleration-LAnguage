
from ogb.nodeproppred import DglNodePropPredDataset
import dgl.data as data
import torch

# DGL default data split comes in masks, while OGB data uses indexes
# Returning masks given the OGB dataset
def ogbDataSplit(dataset):
    split_idx = dataset.get_idx_split()
    train, val, test = torch.zeros((dataset[0][0].num_nodes(),)), torch.zeros((dataset[0][0].num_nodes(),)), torch.zeros((dataset[0][0].num_nodes(),))
    train[split_idx["train"]] = 1
    val[split_idx["valid"]] = 1
    test[split_idx["test"]] = 1
    return torch.BoolTensor(train > 0), torch.BoolTensor(val > 0), torch.BoolTensor(test > 0)


def main(datasetName):
    if datasetName.lower() == "pubmed":
        dataset = data.PubmedGraphDataset()
        graph = dataset[0]
    elif datasetName.lower() == "cora":
        dataset = data.CoraGraphDataset()
        graph = dataset[0]
    elif datasetName.lower() == "citeseer":
        dataset = data.CiteseerGraphDataset()
        graph = dataset[0]
    elif datasetName.lower() == "reddit":
        dataset = data.RedditDataset()
        graph = dataset[0]
    elif datasetName.lower() == "products":
        dataset = DglNodePropPredDataset(name="ogbn-products")
        graph, labelsTransposed = dataset[0]
        graph.ndata["label"] = labelsTransposed.t()[0]
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = ogbDataSplit(dataset)
    elif datasetName.lower() == "arxiv":
        dataset = DglNodePropPredDataset(name="ogbn-arxiv")
        graph, labelsTransposed = dataset[0]
        graph.ndata["label"] = labelsTransposed.t()[0]
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = ogbDataSplit(dataset)
    elif datasetName.lower() == "proteins":
        dataset = DglNodePropPredDataset(name="ogbn-proteins")
        graph, labelsTransposed = dataset[0]
        graph.ndata["label"] = labelsTransposed.t()[0]
        graph.ndata["feat"] = graph.ndata["species"]
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = ogbDataSplit(dataset)
    else:
        raise Exception("Dataset provided is currently not included, currently working on adding more")
    return dataset, graph