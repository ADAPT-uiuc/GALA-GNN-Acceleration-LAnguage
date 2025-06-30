from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected
import torch
import numpy as np
from torch_geometric.data import Data
# from torch_geometric_autoscale import permute
# from collections.abc import Sequence


class PyGLoader:

    def __init__(self, config):
        from ogb.nodeproppred import PygNodePropPredDataset
        if 0:
            # if config.dataset.name in ["products", "papers100M"]:
            self.dataset = PygNodePropPredDataset(
                name=f"ogbn-{config.dataset.name}",
                root=config.dataset.path.strip(config.dataset.name))
            self.data = self.dataset[0]
            if config.dataset.symmetric:
                self.data.edge_index = to_undirected(self.data.edge_index)
            self.split_idx = self.dataset.get_idx_split()
        else:
            basedir = f"../../../../data/{config.dataset.name}/processed/"
            if config.dataset.name in ["twitter", "friendster"]:
                basedir = f"../../../../data/{config.dataset.name}/processed/"
            num_nodes = int(open(basedir + "num_nodes.txt").readline().strip())
            f = open(basedir + "edge_index.dat", "rb")
            edge_index = torch.from_numpy(
                np.fromfile(f, dtype=np.int64).reshape(2, -1))
            if config.dataset.name in ["twitter", "friendster"]:
                x = torch.empty([num_nodes, 384], dtype=torch.float32)
            else:
                x = torch.from_numpy(np.fromfile(
                    open(basedir + "node_features.dat", "rb"), dtype=np.float32).reshape(num_nodes, -1))
            y = torch.from_numpy(np.fromfile(
                open(basedir + "node_labels.dat", "rb"), dtype=np.int64))
            self.data = Data(x=x, edge_index=(
                edge_index[0], edge_index[1]), y=y)
            if config.dataset.symmetric:
                self.data.edge_index = to_undirected(self.data.edge_index)
            self.data.num_nodes = num_nodes
            train_nid = torch.from_numpy(np.fromfile(
                open(basedir + "split/time/train_idx.dat", "rb"), dtype=np.int64))

            # train_nid = np.fromfile(
            #     open(basedir + "split/time/train_idx.dat", "rb"), dtype=np.int64).tolist()
            valid_nid = torch.from_numpy(np.fromfile(
                open(basedir + "split/time/train_idx.dat", "rb"), dtype=np.int64))
            test_nid = torch.from_numpy(np.fromfile(
                open(basedir + "split/time/train_idx.dat", "rb"), dtype=np.int64))
            self.split_idx = {"train": train_nid,
                              "valid": valid_nid, "test": test_nid}

        def transform_wrapper(func):

            def transform(out):
                batch = func(out)
                batch_size = batch.batch_size
                if batch.y.size(0) != batch_size:
                    batch.y = batch.y[:batch_size].flatten().long()
                # batch.to(config.device)
                # batch.mask = torch.arange(batch_size, device=config.device)
                return batch

            return transform
        self.train_loader = NeighborLoader(
            self.data,
            input_nodes=self.split_idx['train'],
            num_neighbors=config.sampler.train.fanouts[::-1],
            shuffle=True,
            batch_size=config.sampler.train.batch_size,
            replace=config.sampler.train.replace)
        self.train_loader.transform_fn = transform_wrapper(
            self.train_loader.transform_fn)
        self.val_loader = NeighborLoader(
            self.data,
            input_nodes=self.split_idx['valid'],
            num_neighbors=config.sampler.eval.fanouts[::-1],
            shuffle=False,
            batch_size=config.sampler.eval.batch_size,
            replace=config.sampler.eval.replace)
        self.val_loader.transform_fn = transform_wrapper(
            self.val_loader.transform_fn)
        self.test_loader = NeighborLoader(
            self.data,
            input_nodes=self.split_idx['test'],
            num_neighbors=config.sampler.eval.fanouts[::-1],
            shuffle=False,
            batch_size=config.sampler.eval.batch_size,
            replace=config.sampler.eval.replace)
        self.test_loader.transform_fn = transform_wrapper(
            self.test_loader.transform_fn)


class GASLoader:

    def __init__(self, config):
        from ogb.nodeproppred import PygNodePropPredDataset
        if 0:
            # if config.dataset.name in ["products", "papers100M"]:
            self.dataset = PygNodePropPredDataset(
                name=f"ogbn-{config.dataset.name}",
                root=config.dataset.path.strip(config.dataset.name))
            self.data = self.dataset[0]
            if config.dataset.symmetric:
                self.data.edge_index = to_undirected(self.data.edge_index)
            self.split_idx = self.dataset.get_idx_split()
        else:
            basedir = f"../../../../data/{config.dataset.name}/processed/"
            if config.dataset.name in ["twitter", "friendster"]:
                basedir = f"../../../../data/{config.dataset.name}/processed/"
            num_nodes = int(open(basedir + "num_nodes.txt").readline().strip())
            f = open(basedir + "edge_index.dat", "rb")
            edge_index = torch.from_numpy(
                np.fromfile(f, dtype=np.int64).reshape(2, -1))
            x = torch.from_numpy(np.fromfile(
                open(basedir + "node_features.dat", "rb"), dtype=np.float32).reshape(num_nodes, -1))
            y = torch.from_numpy(np.fromfile(
                open(basedir + "node_labels.dat", "rb"), dtype=np.int64))
            self.data = Data(x=x, edge_index=(
                edge_index[0], edge_index[1]), y=y)
            if config.dataset.symmetric:
                self.data.edge_index = to_undirected(self.data.edge_index)
            self.data.num_nodes = num_nodes
            train_nid = torch.from_numpy(np.fromfile(
                open(basedir + "split/time/train_idx.dat", "rb"), dtype=np.int64))

            # train_nid = np.fromfile(
            #     open(basedir + "split/time/train_idx.dat", "rb"), dtype=np.int64).tolist()
            valid_nid = torch.from_numpy(np.fromfile(
                open(basedir + "split/time/train_idx.dat", "rb"), dtype=np.int64))
            test_nid = torch.from_numpy(np.fromfile(
                open(basedir + "split/time/train_idx.dat", "rb"), dtype=np.int64))
            self.split_idx = {"train": train_nid,
                              "valid": valid_nid, "test": test_nid}
        ptr = torch.from_numpy(np.fromfile(
            basedir + "cluster_ptr.dat", dtype=np.int64))
        perm = torch.from_numpy(np.fromfile(
            basedir + "cluster.dat", dtype=np.int64))
        assert False
        # self.data = permute(self.data, perm, log=True)

        def transform_wrapper(func):

            def transform(out):
                batch = func(out)
                batch_size = batch.batch_size
                if batch.y.size(0) != batch_size:
                    batch.y = batch.y[:batch_size].flatten().long()
                # batch.to(config.device)
                # batch.mask = torch.arange(batch_size, device=config.device)
                return batch

            return transform
        self.train_loader = NeighborLoader(
            self.data,
            input_nodes=self.split_idx['train'],
            num_neighbors=config.sampler.train.fanouts[::-1],
            shuffle=True,
            batch_size=config.sampler.train.batch_size,
            replace=config.sampler.train.replace)
        self.train_loader.transform_fn = transform_wrapper(
            self.train_loader.transform_fn)
        self.val_loader = NeighborLoader(
            self.data,
            input_nodes=self.split_idx['valid'],
            num_neighbors=config.sampler.eval.fanouts[::-1],
            shuffle=False,
            batch_size=config.sampler.eval.batch_size,
            replace=config.sampler.eval.replace)
        self.val_loader.transform_fn = transform_wrapper(
            self.val_loader.transform_fn)
        self.test_loader = NeighborLoader(
            self.data,
            input_nodes=self.split_idx['test'],
            num_neighbors=config.sampler.eval.fanouts[::-1],
            shuffle=False,
            batch_size=config.sampler.eval.batch_size,
            replace=config.sampler.eval.replace)
        self.test_loader.transform_fn = transform_wrapper(
            self.test_loader.transform_fn)
