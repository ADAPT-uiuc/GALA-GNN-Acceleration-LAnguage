from models.dgl_models import *
from models.pyg_models import *
try:
    from dataLoading import dgl_loadData, pyg_loadData
except ImportError:
    print("Warning: Failed to Import modules from loadData files")
from training import dgl_train, pyg_train
import os
os.environ["DGLBACKEND"] = "pytorch"
import torch
import torch.nn as nn
import torch.nn.functional as F
from sys import exit, argv
import csv

# TODO - use replace and split functions once in beginning, this is too messy 
def error_check(benchmark):
    args = benchmark.replace(", ", ",").replace("\n","").split()
    if (len(args) <= 4):
        print(benchmark.replace(", ", ",").split())
        print("Incorrect data format, please try again")
        return 0
    if (benchmark.rstrip().split()[1] == "gat" and benchmark.rstrip()[-1] != "]"):
        print("Graph Attention Networks must have array of heads for input and hidden layers, please try again")
        return 0
    if (args[0].lower() == "graphiler"):
        if (args[1].lower() not in ["gcn", "gat"]):
            print("Graphiler only supports GCNs and GATs")
            return 0
        if (args[3].lower() not in ["cora", "pubmed", "arxiv", "reddit"]):
            print("Graphiler only supports Cora, PubMed, Arxiv, and Reddit")
            return 0
    return 1

def main(benchmark):
    benchmark = benchmark.replace(", ", ",").split()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if benchmark[0].lower() == "dgl":
        dataset, graph = dgl_loadData.main(benchmark[3])
        graph = graph.to(device)

        # converting to int array, adding feature size and embedding size
        sizes = [int(x) for x in benchmark[2].strip("[").strip("]").replace(" ", "").split(",")]
        sizes.insert(0, graph.ndata["feat"].shape[1])
        sizes.append(dataset.num_classes)

        if benchmark[1].lower() == "gcn":
            model = dgl_GCN(sizes[0], sizes[1:-1], sizes[-1]).to(device)
        elif benchmark[1].lower() == "gat":
            heads = [int(x) for x in benchmark[5].strip("[").strip("]").replace(" ", "").split(",")]
            model = dgl_GAT(sizes[0], sizes[1:-1], sizes[-1], heads).to(device)
        elif benchmark[1].lower() == "sage":
            model = dgl_SAGE(sizes[0], sizes[1:-1], sizes[-1]).to(device)
        elif benchmark[1].lower() == "gin":
            model = dgl_GIN(sizes[0], sizes[1:-1], sizes[-1]).to(device)
        return dgl_train.train(graph, model, int(benchmark[4]))

    elif benchmark[0].lower() == "pyg":
        dataset, graph = pyg_loadData.main(benchmark[3])
        graph = graph.to(device)

        # converting to int array, adding feature size and embedding size
        sizes = [int(x) for x in benchmark[2].strip("[").strip("]").replace(" ", "").split(",")]
        sizes.insert(0, dataset.num_node_features)
        sizes.append(dataset.num_classes)

        if benchmark[1].lower() == "gcn":
            model = pyg_GCN(sizes[0], sizes[1:-1], sizes[-1]).to(device)
        elif benchmark[1].lower() == "gat":
            heads = [int(x) for x in benchmark[5].strip("[").strip("]").replace(" ", "").split(",")]
            model = pyg_GAT(sizes[0], sizes[1:-1], sizes[-1], heads).to(device)
        elif benchmark[1].lower() == "sage":
            model = pyg_SAGE(sizes[0], sizes[1:-1], sizes[-1]).to(device)
        elif benchmark[1].lower() == "gin":
            model = pyg_GIN(sizes[0], sizes[1:-1], sizes[-1]).to(device)
        return pyg_train.train(graph, model, int(benchmark[4]))
    
    elif benchmark[0].lower() == "graphiler":
        if benchmark[1].lower() == "gcn":
            os.system("python graphiler/examples/GAT/GAT.py " + benchmark[3] + " " + benchmark[4])
        elif benchmark[1].lower() == "gat":
            os.system("python graphiler/examples/GCN/GCN.py " + benchmark[3] + " " + benchmark[4])


if __name__ == "__main__":
    input = argv[1]
    if input.endswith(".txt"):
        f = open(input, "r")
        for x in f:
            if error_check(x) == 0:
                exit()
            main(x)
    else:
        raise Exception("Input must be file with .txt")
