import matplotlib.pyplot as plt
import copy
import torch
import numpy
import cxgnndl_backend
import sys
datadir = "../../../../data/{}/processed/{}"
dset = sys.argv[1]
if dset in ["twitter", "friendster", "rmag240m"]:
    datadir = "../../../../data/{}/processed/{}"
ptr = numpy.fromfile(datadir.format(
    dset, "csr_ptr_undirected.dat"), dtype=numpy.int64)
idx = numpy.fromfile(datadir.format(
    dset, "csr_idx_undirected.dat"), dtype=numpy.int64)
train_nid = torch.from_numpy(numpy.fromfile(datadir.format(
    dset, "split/time/train_idx.dat"), dtype=numpy.int64))
num_node = int(open(datadir.format(dset, "num_nodes.txt")).readline().strip())
print(num_node)
ptr = torch.from_numpy(ptr)
idx = torch.from_numpy(idx)

print("ptr {} idx {}".format(ptr.shape, idx.shape))
deg = ptr[1:] - ptr[:-1]
print("deg {}".format(deg.float().mean()))
print("deg {}".format(deg.float()[train_nid].mean()))


outputs = cxgnndl_backend.graph_analysis(
    ptr, idx, list(train_nid.numpy()), num_node, 3)
for item in outputs:
    print(item.max(), item.min(), item.float().mean(), item.float().std())
for i in range(len(outputs)):
    outputs[i], indices = outputs[i].sort(dim=0, descending=True)
    outputs[i] = outputs[i][outputs[i] != 0]
    torch.cumsum(outputs[i], dim=0, out=outputs[i])
    mmax = outputs[i][-1].item()
    # outputs[i] = outputs[i].float() / outputs[-1]
    plt.plot(outputs[i].numpy() / mmax, label="{}-{}-{}".format(dset, i, mmax))
    print(mmax / outputs[i].shape[0])
    # plt.clf()
plt.legend()
plt.savefig("{}.pdf".format(dset))
# visit_map = torch.zeros(num_node, dtype=torch.bool)
# visit_map[train_nid] = True
# num_layer = 3
# for i in range(num_layer):
#     new_train_nid = list(copy.deepcopy(train_nid))
#     for item in train_nid:
#         for j in range(ptr[item], ptr[item+1]):
#             if not visit_map[idx[j]]:
#                 new_train_nid.append(idx[j])
#                 visit_map[idx[j]] = True
#     print(len(new_train_nid))
#     train_nid = copy.deepcopy(new_train_nid)
