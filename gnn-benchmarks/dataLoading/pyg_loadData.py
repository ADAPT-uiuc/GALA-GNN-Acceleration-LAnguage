import torch
import torch_geometric.datasets as data
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import scatter

dataSetDirectory = "./pyg/datasets"


# DGL default data split comes in masks, while OGB data uses indexes
# Returning masks given the OGB dataset
def ogbDataSplit(dataset):
    split_idx = dataset.get_idx_split()
    train, val, test = torch.zeros((dataset[0].num_nodes,)), torch.zeros((dataset[0].num_nodes,)), torch.zeros((dataset[0].num_nodes,))
    train[split_idx["train"]] = 1
    val[split_idx["valid"]] = 1
    test[split_idx["test"]] = 1
    return torch.BoolTensor(train > 0), torch.BoolTensor(val > 0), torch.BoolTensor(test > 0)

def main(datasetName):
    transform = T.Compose([T.RandomNodeSplit("random")])
    if datasetName.lower() == "pubmed":
        dataset = data.Planetoid(root=dataSetDirectory, name="PubMed", transform=transform)
        graph = dataset[0]
    elif datasetName.lower() == "cora":
        dataset = data.Planetoid(root=dataSetDirectory, name="Cora", transform=transform)
        graph = dataset[0]
    elif datasetName.lower() == "citeseer":
        dataset = data.Planetoid(root=dataSetDirectory, name="CiteSeer", transform=transform)
        graph = dataset[0]
    elif datasetName.lower() == "reddit":
        dataset = data.Reddit()
        graph = dataset[0]
    elif datasetName.lower() == "products":
        dataset = PygNodePropPredDataset(name="ogbn-products")
        graph = dataset[0]
        graph.y = graph.y.t()[0]
        graph.train_mask, graph.val_mask, graph.test_mask = ogbDataSplit(dataset)
    elif datasetName.lower() == "arxiv":
        dataset = PygNodePropPredDataset(name="ogbn-arxiv")
        graph = dataset[0]
        graph.y = graph.y.t()[0]
        graph.train_mask, graph.val_mask, graph.test_mask = ogbDataSplit(dataset)
    elif datasetName.lower() == "proteins":
        dataset = PygNodePropPredDataset(name="ogbn-proteins")
        graph = dataset[0]
        row, col = graph.edge_index
        graph.x = scatter(graph.edge_attr, col, dim_size=graph.num_nodes, reduce='sum')
        graph.y = graph.y.t()[0]
        graph.train_mask, graph.val_mask, graph.test_mask = ogbDataSplit(dataset)

    else:
        raise Exception("Dataset provided is currently not included, currently working on adding more")
    return dataset, graph