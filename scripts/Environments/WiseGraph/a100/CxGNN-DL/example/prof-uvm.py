import cxgnndl_backend
import time
import torch
a = torch.empty([1000000, 512], dtype=torch.float32, pin_memory=True)

for i in range(100):
    b = torch.randint(1000000, [10000]).cuda()
    torch.cuda.synchronize()
    t0 = time.time()
    cxgnndl_backend.uvm_select(a, b)
    torch.cuda.synchronize()
    print("bandwidth", 10000 * 512 * 4 /
          (time.time() - t0) / 1024 / 1024 / 1024)
