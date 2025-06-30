import torch
import os
import cxgnncomp as cxgc
from cxgnncomp import (
    MySageConv,
    MyGATConv,
    MyRGCNConv,
    MyGCNConv,
    LSTMConv,
    get_conv_from_str,
    get_model_from_str,
    Batch,
    PyGBatch,
    reorder_by,
)
import dgl
import time
import subprocess
import numpy as np

def train_epoch(model, params, label, optimizer, lossfn):
    for epoch in range(1, 100):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.time()
        
        out = model(*params)
        if isinstance(model, LSTMConv):
            return
        
        torch.cuda.synchronize()
        t1 = time.time()
        
        loss = lossfn(out, label)
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        t2 = time.time()
        
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # in MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**2)  # in MB
        
        gpu_stats = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True).stdout.strip()
        
        gpu_used, gpu_total = map(float, gpu_stats.split(","))
        
        _, predicted = torch.max(out, 1)
        correct = (predicted == label).sum().item()
        accuracy = correct / label.size(0) * 100
        
        #print(f"forward {t1 - t0} backward {t2 - t1}")
        #print(f"Epoch {epoch} - Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} MB, Reserved: {gpu_memory_reserved:.2f} MB")
        print(f"GPU Used/Total (nvidia-smi): {gpu_used:.2f}/{gpu_total:.2f} MB")
        
def train_epoch_perf(model, params, label, optimizer, lossfn):
    total_loss = 0.0
    total_allocated = 0.0
    total_reserved = 0.0
    total_gpu_used = 0.0
    total_gpu_total = 0.0
    num_epochs = 99

    for epoch in range(1, 100):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.time()

        out = model(*params)
        if isinstance(model, LSTMConv):
            return

        torch.cuda.synchronize()
        t1 = time.time()

        loss = lossfn(out, label)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t2 = time.time()

        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**2)    # MB

        gpu_stats = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip()
        gpu_used, gpu_total = map(float, gpu_stats.split(","))

        _, predicted = torch.max(out, 1)
        correct = (predicted == label).sum().item()
        accuracy = correct / label.size(0) * 100

        total_loss += loss.item()
        total_allocated += gpu_memory_allocated
        total_reserved += gpu_memory_reserved
        total_gpu_used += gpu_used
        total_gpu_total += gpu_total

    print(f"Average GPU Memory Allocated: {total_allocated / num_epochs:.2f} MB")
    print(f"Average GPU Memory Reserved: {total_reserved / num_epochs:.2f} MB")
    print(f"Average GPU Used/Total (nvidia-smi): {total_gpu_used / num_epochs:.2f}/{total_gpu_total / num_epochs:.2f} MB")

def train(model, params, label, optimizer, lossfn):
    torch.cuda.synchronize()
    t0 = time.time()
    optimizer.zero_grad()
    out = model(*params)
    if isinstance(model, LSTMConv):
        return
    torch.cuda.synchronize()
    t1 = time.time()
    loss = lossfn(out, label)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t2 = time.time()
    timer = cxgc.get_timers()
    #print(f"forward {t1 - t0} backward {t2 - t1}")
    print(f"Inference Time: {t1 - t0} Training Time {t1 - t0 + t2 - t1}")
    # timer.log_all(print)


def test_conv_training(args):
    infeat = args.infeat
    outfeat = args.outfeat
    num_head = args.num_head
    dev = torch.device("cuda:0")
    dset = args.dataset
    is_full_graph = args.is_full_graph

    feat, ptr, idx, b = cxgc.prepare_graph(
        dset=dset, feat_len=infeat, num_head=num_head, num_seeds=args.num_seeds
    )
    feat_label = torch.randn(
        [ptr.shape[0] - 1, outfeat], dtype=torch.float32, device=dev
    )
    feat.requires_grad_(True)

    cxgc.set_timers()

    conv = get_conv_from_str(args.model, infeat, outfeat, num_head).to(dev)
    conv.reset_parameters()
    optimizer = torch.optim.Adam(conv.parameters(), lr=0.1)
    lossfn = torch.nn.MSELoss()
    torch.cuda.synchronize()

    if isinstance(conv, MyGATConv):
        cxgc.prof(
            "train conv",
            args.model,
            lambda: train(
                conv,
                [
                    feat,
                    ptr,
                    idx,
                    b["num_node_in_layer"][-2],
                    b["num_node_in_layer"][-1],
                    idx.shape[0],
                ],
                feat_label,
                optimizer,
                lossfn,
            ),
        )
    elif isinstance(conv, MyRGCNConv):
        edge_types = torch.randint(0, conv.num_rel, (idx.shape[0],), device=dev)
        cxgc.prof(
            "train conv",
            args.model,
            lambda: train(
                conv,
                [feat, ptr, idx, edge_types, b["num_node_in_layer"][-2]],
                feat_label,
                optimizer,
                lossfn,
            ),
        )
    else:
        cxgc.prof(
            "train conv",
            args.model,
            lambda: train(
                conv,
                [feat, ptr, idx, b["num_node_in_layer"][-2]],
                feat_label,
                optimizer,
                lossfn,
            ),
        )
    torch.cuda.synchronize()


def get_dset_config(dset):
    if "arxiv" in dset:
        infeat = 128
        outfeat = 64
    elif "products" in dset:
        infeat = 100
        outfeat = 64
    elif dset == "reddit":
        infeat = 602
        outfeat = 41
    elif "paper" in dset:
        infeat = 128
        outfeat = 256
    elif "friendster" in dset:
        infeat = 384
        outfeat = 64
    elif "pubmed" in dset:
        infeat = 500
        outfeat = 32
    elif "cora" == dset:
        infeat = 1433
        outfeat = 32
    elif "corafull" == dset:
        infeat = 8710
        outfeat = 96
    elif "ogbn-papers100M_1" in dset:
        infeat = 128
        outfeat = 172
    elif "ogbn-papers100M_2" in dset:
        infeat = 128
        outfeat = 172
    elif "ogbn-papers100M_5" in dset:
        infeat = 128
        outfeat = 172
    elif "ogbn-papers100M_10" in dset:
        infeat = 128
        outfeat = 172
    elif "ogbn-papers100M_20" in dset:
        infeat = 128
        outfeat = 172
    elif "ogbn-papers100M_25" in dset:
        infeat = 128
        outfeat = 172
    else:
        assert False, "unknown dataset"
    return infeat, outfeat


def get_model(args):
    mtype = args.model
    graph_type = args.graph_type
    hiddenfeat = args.hidden_feat
    num_layer = args.num_layer
    infeat, outfeat = get_dset_config(args.dataset)
    if mtype.upper() == "LSTM":
        infeat, outfeat = args.dedicate_feat, args.dedicate_feat
    dev = "cuda"
    num_head = args.num_head
    model = get_model_from_str(
        mtype,
        infeat,
        hiddenfeat,
        outfeat,
        graph_type,
        num_layer,
        num_head,
        args.num_rel,
        args.dataset,
        dropout=0,
    ).to(dev)
    return model


def to_dgl_block(ptr, idx, num_node_in_layer, num_edge_in_layer):
    print("num_node_in_layer", num_node_in_layer, num_edge_in_layer)
    blocks = []
    num_layer = num_node_in_layer.shape[0] - 1
    for i in range(len(num_node_in_layer) - 1):
        num_src = num_node_in_layer[num_layer - i]
        num_dst = num_node_in_layer[num_layer - i - 1]
        ptr = ptr[: num_dst + 1]
        idx = idx[: num_edge_in_layer[num_layer - i - 1]]
        blocks.append(
            dgl.create_block(
                ("csc", (ptr, idx, torch.tensor([]))), int(num_src), int(num_dst)
            )
        )
    return blocks

def load_labels(dset, num_nodes, device):
    """Load node labels from the file."""
    current_path = os.path.dirname(os.path.realpath(__file__))
    label_file = os.path.abspath(os.path.join(current_path, f"../../../../../data/{dset}/processed/node_labels.dat"))
    #print(f"Loading labels from {label_file}")
    labels = np.fromfile(label_file, dtype=np.int64)
    sub_labels_np = labels.flatten()
    #print("Unique labels in subgraph:", np.unique(sub_labels_np))
    #print("Min label:", labels.min().item(), "Max label:", labels.max().item())
    min_label = labels.min()
    labels[labels == min_label] = 1
    if labels.shape[0] != num_nodes:
        raise ValueError(f"Expected {num_nodes} labels but found {labels.shape[0]} in {label_file}")
    return torch.from_numpy(labels).to(device)

def run_model(args, model):
    dset = args.dataset
    if (
        args.model.upper() in ["SAGE", "GCN"]
        and args.graph_type == "CSR_Layer"
        and dset == "arxiv"
    ):
        dset += "-ng"
    infeat, outfeat = get_dset_config(args.dataset)
    print("Dataset: ", dset)
    if args.model.upper() == "LSTM":
        infeat, outfeat = args.dedicate_feat, args.dedicate_feat
    num_head = args.num_head
    dev = torch.device("cuda:0")
    feat, ptr, idx, b, edge_index = cxgc.prepare_graph(
        dset=dset,
        feat_len=infeat,
        num_head=num_head,
        num_seeds=args.num_seeds,
        is_full_graph=args.is_full_graph,
        need_edge_index=True,
        num_layer=args.num_layer,
    )
    if args.graph_type == "CSR_Layer":
        edge_index = None
        if args.model == "LSTM":
            deg = ptr[1:] - ptr[:-1]
            metric = torch.argsort(deg, descending=False)
            ptr, idx = reorder_by(ptr, idx, metric)
    feat_label = torch.randn(
        [b["num_node_in_layer"][0], outfeat], dtype=torch.float32, device=dev
    )
    #print(feat_label.shape)
    num_nodes = b["num_node_in_layer"][0]
    feat_label = load_labels(dset, num_nodes, dev)
    #print(feat_label.shape)
    num_classes = len(torch.unique(feat_label))
    #print(f"Number of unique classes in labels: {num_classes}")
    # NG
    # new_ptr, new_target = cxgc.neighbor_grouping(ptr, neighbor_thres=32)
    # feat = torch.randn([new_ptr.shape[0] - 1, infeat],device=feat.device)
    # feat_label = torch.randn([new_ptr.shape[0] - 1, outfeat],device=feat.device)
    # ptr = new_ptr
    # b["num_node_in_layer"] = [ptr.shape[0] - 1 for i in range(len(b["num_node_in_layer"]))]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    lossfn = torch.nn.CrossEntropyLoss()
    if args.graph_type == "CSR_Layer" or args.model == "LSTM":
        #train_epoch(model,[Batch(feat,ptr,idx,b["num_node_in_layer"],b["num_edge_in_layer"],edge_index=edge_index,)],feat_label,optimizer,lossfn,)
        output = cxgc.prof(
            args.graph_type,
            args.model,
            lambda: train(
                model,
                [
                    Batch(
                        feat,
                        ptr,
                        idx,
                        b["num_node_in_layer"],
                        b["num_edge_in_layer"],
                        edge_index=edge_index,
                    )
                ],
                feat_label,
                optimizer,
                lossfn,
            ),
        )
        train_epoch_perf(model,[Batch(feat,ptr,idx,b["num_node_in_layer"],b["num_edge_in_layer"],edge_index=edge_index,)],feat_label,optimizer,lossfn,)
    elif args.graph_type == "DGL":
        dgl_blocks = to_dgl_block(
            ptr, idx, b["num_node_in_layer"], b["num_edge_in_layer"]
        )
        output = cxgc.prof(
            args.graph_type,
            args.model,
            lambda: train(model, [[dgl_blocks, feat]], feat_label, optimizer, lossfn),
        )
    elif args.graph_type == "COO" or args.graph_type == "PyG":
        output = cxgc.prof(
            args.graph_type,
            args.model,
            lambda: train(
                model, [PyGBatch(feat, edge_index)], feat_label, optimizer, lossfn
            ),
        )
    else:
        assert False, "unknown graph type"
    #print(f"ans {args.dataset} {args.model} {args.graph_type} {output}")

    cxgc.global_tuner.save()
    
    print("==========================================")
 
def test_model_training(args):
    cxgc.global_tuner.set_lazy(lazy=False)
    cxgc.set_timers()
    model = get_model(args)
    run_model(args, model)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--graph_type", type=str, default="CSR_Layer")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--hidden_feat", type=int, default=1024)
    parser.add_argument("--num_layer", type=int, default=8)
    parser.add_argument("--num_head", type=int, default=1)
    parser.add_argument("--num_rel", type=int, default=7)
    parser.add_argument("--infeat", type=int, default=-1)
    parser.add_argument("--outfeat", type=int, default=-1)
    parser.add_argument("--is_full_graph", type=int, default=1)
    parser.add_argument("--num_seeds", type=int, default=1000)
    parser.add_argument("--dedicate_feat", type=int, default=32)
    args = parser.parse_args()
    print(args)
    if args.infeat > 0 and args.outfeat > 0:
        print("Benchmark single conv")
        test_conv_training(args)
    else:
        print("Benchmark model inference & training")
        test_model_training(args)
