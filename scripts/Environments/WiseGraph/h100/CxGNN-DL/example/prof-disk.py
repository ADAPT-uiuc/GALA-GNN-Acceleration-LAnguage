import torch
import time
import numpy as np
import cxgnndl_backend
import sys

datadir = "../../../../data/mag240m/processed/node_features.dat"
num_nodes = int(open(
    "../../../../data/mag240m/processed/num_nodes.txt").readline().strip())
feature_len = 384
method = int(sys.argv[1])
mmap = cxgnndl_backend.gen_mmap(datadir, feature_len, 32, method == 3)
for iter in range(100):
    index = torch.randint(num_nodes, [10000])
    start = time.time()
    if method == 1:
        cxgnndl_backend.single_thread_mmap_load(mmap, index)
    elif method == 2:
        cxgnndl_backend.mmap_select(mmap, index)
    elif method == 3:
        cxgnndl_backend.mmap_select(mmap, index)
    print("bandwidth", 10000 * 384 * 4 /
          (time.time() - start) / 1024 / 1024 / 1024)
