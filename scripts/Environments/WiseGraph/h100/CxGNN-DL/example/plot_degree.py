import matplotlib.pyplot as plt
import numpy
import torch
dsets = ["twitter", "friendster", "rmag240m", "mag240m", "papers100M"]
mmax_num_node = 0
for dset in dsets:
    datadir = "../../../../data/{}/processed/{}"
    if dset in ["twitter", "friendster", "rmag240m"]:
        datadir = "../../../../data/{}/processed/{}"
    num_node = int(
        open(datadir.format(dset, "num_nodes.txt")).readline().strip())
    mmax_num_node = max(mmax_num_node, num_node)
for dset in dsets:
    datadir = "../../../../data/{}/processed/{}"
    if dset in ["twitter", "friendster", "rmag240m"]:
        datadir = "../../../../data/{}/processed/{}"
    num_node = int(
        open(datadir.format(dset, "num_nodes.txt")).readline().strip())
    ptr = numpy.fromfile(datadir.format(
        dset, "csr_ptr_undirected.dat"), dtype=numpy.int64)
    deg = ptr[1:] - ptr[:-1]
    deg = torch.from_numpy(deg)
    # deg = torch.bincount(deg)
    deg, indices = deg.sort(dim=0, descending=True)
    deg = torch.cumsum(deg, dim=0)
    mmax = deg[-1].item()
    x = torch.arange(deg.shape[0]).numpy()
    plt.plot(x / num_node * mmax_num_node,  deg.numpy() /
             mmax, label="{}-{}".format(dset, mmax))
plt.legend()
plt.savefig("degree.pdf")
