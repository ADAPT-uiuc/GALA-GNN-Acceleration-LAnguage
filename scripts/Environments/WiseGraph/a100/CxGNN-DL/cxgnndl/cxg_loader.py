import cxgnndl_backend
from .feature_data import UVM
import numpy as np
import os

def load_full_graph_structure(name, undirected=True):
    current_path = os.path.dirname(os.path.realpath(__file__))
    basedir = os.path.abspath(os.path.join(current_path, "../../../data/"))
    print("current_path:", current_path)
    print("resolved basedir:", basedir)
    if undirected:
        prop = "undirected"
    else:
        prop = "directed"
    print(basedir)
    print(os.path.join(basedir, name, "processed/"))
    ptr_path = os.path.join(basedir, name, f"processed/csr_ptr_{prop}.dat")
    idx_path = os.path.join(basedir, name, f"processed/csr_idx_{prop}.dat")
    ptr = np.fromfile(ptr_path, dtype=np.int64)
    idx = np.fromfile(idx_path, dtype=np.int64)
    return ptr, idx


class CXGLoader:

    def __init__(self, config):
        self.backend = cxgnndl_backend.CXGDL("new_config.yaml")
        self.split = 'train'
        self.feat_mode = config["loading"]["feat_mode"]
        print("CXGLoader feat_mode", self.feat_mode)
        if self.feat_mode in ["mmap", "uvm", "random"]:
            self.uvm = UVM(config)
        else:
            self.uvm = None
        # self.uvm = None

    def __len__(self):
        return self.backend.num_iters()

    def __iter__(self):
        return self

    def __next__(self) -> cxgnndl_backend.Batch:
        batch = self.backend.get_batch()
        if not self.uvm is None:
            batch.x = self.uvm.get(batch.sub_to_full)
        return batch

    @property
    def train_loader(self):
        self.split = 'train'
        self.backend.start(self.split)
        return self

    @property
    def val_loader(self):
        self.split = 'valid'
        self.backend.start(self.split)
        return self

    @property
    def test_loader(self):
        self.split = 'test'
        self.backend.start(self.split)
        return self
