import graph_loader_backend
import time
import torch

num_device = 4
num_nodes = 1000000
feat = 512 // 4
num_trial = 10


output = [None for i in range(num_device)]
for iter in range(num_trial):
    arr = []
    for i in range(num_device):
        for j in range(num_device):
            arr.append(torch.randn([num_nodes, feat],
                                   dtype=torch.float32, device=j) * (i * num_device + j))
    for j in range(num_device):
        torch.cuda.synchronize(j)
    t0 = time.time()
    for tar_it in range(num_device):
        output[tar_it] = torch.nn.parallel.comm.gather(
            arr[tar_it * num_device: tar_it * num_device + num_device],
            destination=tar_it,
            dim=1
        )
    for j in range(num_device):
        torch.cuda.synchronize(j)
    print("bandwidth1", num_nodes * feat * 4 /
          (time.time() - t0) / 1024 / 1024 / 1024 * 12)

for i in range(num_trial):
    t0 = time.time()
    output = graph_loader_backend.mygather2(arr, num_device)
    for j in range(num_device):
        torch.cuda.synchronize(j)
    print("bandwidth2", num_nodes * feat * 4 /
          (time.time() - t0) / 1024 / 1024 / 1024 * 12)


for i in range(num_trial):
    t0 = time.time()
    output = graph_loader_backend.mygather(arr, num_device)
    for j in range(num_device):
        torch.cuda.synchronize(j)
    print("bandwidth3", num_nodes * feat * 4 /
          (time.time() - t0) / 1024 / 1024 / 1024 * 12)
