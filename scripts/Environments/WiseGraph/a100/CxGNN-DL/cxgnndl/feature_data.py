import torch
import cxgnndl_backend
from os import path
import numpy as np
from .util import log


class UVM:

    def __init__(self, config):
        num_node_datapath = path.join(str(config["dataset"]["path"]),
                                      "processed", "num_nodes.txt")
        self.num_node = int(open(num_node_datapath).readline())
        self.in_channel = int(config["dataset"]["feature_dim"])
        buffer_path = path.join(str(config["dataset"]["path"]), "processed",
                                "node_features.dat")
        dset_name = config.dataset.name.lower()
        if dset_name in ["mag240m", "rmag240m"]:
            self.data_type = np.float16
            self.torch_type = torch.float16
        else:
            self.data_type = np.float32
            self.torch_type = torch.float32
        self.mode = config["loading"]["feat_mode"]
        self.dry_run = config.performance.get("dry_run", False)
        if not self.dry_run:
            if self.mode == "uvm":
                print("==== Using memory ====")
                # if "twitter" in buffer_path or "friendster" in buffer_path or "mag240_384" in buffer_path or "mag240_768" in buffer_path:
                random_dsets = ["twitter", "friendster",
                                "mag240m_384", "mag240m_768", "rmag240m_384", "rmag240m_768"]
                if dset_name in random_dsets:
                # if 1:
                    log.warn(f"EMTPY for {random_dsets}")
                    self.buffer = torch.randn(
                        [self.num_node, self.in_channel],
                        dtype=self.torch_type, pin_memory=True)
                elif dset_name == "rmag240m":
                    self.buffer = torch.empty([self.num_node, self.in_channel],
                                              dtype=torch.float16,
                                              pin_memory=True)
                    cxgnndl_backend.read_to_ptr(
                        self.buffer.data_ptr(), buffer_path,
                        self.num_node * self.in_channel * 2)
                else:
                    arr = np.fromfile(buffer_path, dtype=self.data_type)
                    arr = arr.reshape([-1, self.in_channel])
                    print(arr.shape, self.num_node, self.in_channel)
                    assert (arr.shape[0] == self.num_node
                            and arr.shape[1] == self.in_channel)
                    self.buffer = torch.from_numpy(arr).pin_memory()
            elif self.mode == "mmap":
                print("==== Using MMAP ====")
                # if "twitter" in buffer_path or "friendster" in buffer_path or "arxiv" in buffer_path:
                if 1:
                    log.warn(f"EMTPY for {buffer_path}")
                    buffer_path = "../../../../data/rmag240m/processed/node_features.dat"
                self.buffer = cxgnndl_backend.gen_mmap(
                    buffer_path, self.in_channel,
                    32 if self.data_type == np.float32 else 16, config["mmap"]["random"])
                print(self.buffer.shape)
            elif self.mode == "random":
                pass
            else:
                assert False, "Unknown mode: " + self.mode
            print("==== Finish Loading ====")
        self.device = torch.device(config["device"])
        self.has_cache = False
        if float(config.performance.get("cache_size", 0)) > 0:
            cs = config.performance.get("cache_size", 0)
            if cs > 1:
                cs = cs / self.num_node
            self.init_cache(config, cs)
        self.hit_rate_arr = []

    def init_cache(self, config, percent=0.1):
        ptr_datapath = path.join(str(config["dataset"]["path"]), "processed",
                                 "csr_ptr_undirected.dat")
        ptr = torch.from_numpy(np.fromfile(ptr_datapath,
                                           dtype=np.int64)).to(self.device)
        self.cache_map = torch.zeros(self.num_node,
                                     dtype=torch.bool,
                                     device=self.device)
        deg = ptr[1:] - ptr[:-1]  # in degree
        del ptr
        sorted, indice = torch.sort(deg, descending=True)
        del sorted
        del deg
        cache_indice = indice[:int(percent * self.num_node)]
        self.cache_map[cache_indice] = True
        new_id = torch.arange(0, len(cache_indice), device=self.device)
        self.full2cache = torch.ones(self.num_node,
                                     device=self.device).long().neg()
        self.full2cache = self.full2cache.index_put_(
            [cache_indice], new_id)  # [-1, 0, -1, 1, -1, 2]
        if not self.dry_run:
            self.cache = self.buffer[cache_indice].to(self.device)
        self.has_cache = True

    def get(self, index):
        # torch.cuda.synchronize()
        # t0 = time.time()
        if self.mode == "uvm":
            if self.data_type == np.float32:
                output = cxgnndl_backend.uvm_select(self.buffer, index)
            else:
                output = cxgnndl_backend.uvm_select_half(self.buffer,
                                                         index).float()
        elif self.mode == "mmap":
            output = cxgnndl_backend.mmap_select(self.buffer,
                                                 index.cpu()).to(self.device).float()
        elif self.mode == "random":
            output = torch.randn([index.shape[0], self.in_channel],
                                 device=self.device)
        else:
            assert False, "Unknown mode: " + self.mode
        # torch.cuda.synchronize()
        # print(time.time() - t0, output.shape)
        return output

    def masked_get(self, index, mask):
        # torch.cuda.synchronize()
        # t0 = time.time()
        if self.dry_run:
            output = torch.randn([index.shape[0], self.in_channel],
                                 device=self.device)
        elif self.mode == "uvm":
            if self.data_type == np.float32:
                output = cxgnndl_backend.uvm_select_masked(
                    self.buffer, index, mask)
            else:
                output = cxgnndl_backend.uvm_select_masked_half(
                    self.buffer, index, mask).float()
        elif self.mode == "mmap":
            output = cxgnndl_backend.mmap_select(
                self.buffer,
                index.masked_fill_(~(mask.bool()), 0).cpu()).to(self.device).float()
        else:
            assert False, "Unknown mode: " + self.mode
        # torch.cuda.synchronize()
        # print(time.time() - t0, output.shape)
        return output

    def cached_masked_get(self, index, mask):
        mem_access_mask = self.cache_map[index]
        uvm_mask = mask & (~mem_access_mask)
        cache_mask = mask & mem_access_mask
        cache_index = cache_mask.nonzero().view(-1)
        if self.dry_run:
            output = torch.randn([index.shape[0], self.in_channel],
                                 device=self.device)
        else:
            output = self.masked_get(index, uvm_mask)
            output = output.index_put_(
                [cache_index],
                torch.index_select(self.cache, 0,
                                   self.full2cache[index[cache_index]]))
        log.info(
            f"from-cache: {cache_index.shape[0]} from-uvm: {torch.sum(uvm_mask)} needed: {torch.sum(mask)} in-cache: {torch.sum(mem_access_mask)} total: {index.shape[0]} cache_in_use: {cache_index.shape[0] / torch.sum(mask)} cache_in_prune {(torch.sum(mem_access_mask) - cache_index.shape[0]) / (index.shape[0] - torch.sum(mask))}"
        )
        log.info(
            f"pure-cache: {1 - torch.sum(mem_access_mask)/index.shape[0]} pure-history: {torch.sum(mask)/index.shape[0]} combined: {torch.sum(uvm_mask)/index.shape[0]}"
        )
        self.hit_rate_arr.append(torch.sum(uvm_mask) / index.shape[0])
        if len(self.hit_rate_arr) == 1000:
            log.info(
                f"ans: {sum(self.hit_rate_arr[100:])/len(self.hit_rate_arr[100:])}"
            )
            exit()
        return output


def get_uvm(config):
    return UVM(config)
