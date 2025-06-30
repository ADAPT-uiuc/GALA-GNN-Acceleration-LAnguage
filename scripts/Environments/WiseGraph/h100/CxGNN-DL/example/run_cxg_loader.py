import time
import dgl
import cxgnndl
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch

log = logging.getLogger(__name__)


def dump(batch):
    out_dict = {}
    out_dict["num_node_in_layer"] = batch.num_node_in_layer
    out_dict["num_edge_in_layer"] = batch.num_edge_in_layer
    out_dict["ptr"] = batch.ptr.cpu()
    out_dict["idx"] = batch.idx.cpu()
    torch.save(out_dict, "dump.pt")


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device).long()
    return batch_inputs, batch_labels


def load_subtensor_uvm(nfeat, labels, seeds, input_nodes, device):
    input_nodes = input_nodes.to(device)
    batch_inputs = nfeat[input_nodes]
    batch_labels = labels[seeds].to(device).long()
    return batch_inputs, batch_labels


def to_dgl_block(batch):
    blocks = []
    num_layer = batch.num_node_in_layer.shape[0] - 1
    for i in range(len(batch.num_node_in_layer) - 1):
        num_src = batch.num_node_in_layer[num_layer - i]
        num_dst = batch.num_node_in_layer[num_layer - i - 1]
        ptr = batch.ptr[:num_dst + 1]
        idx = batch.idx[:batch.num_edge_in_layer[num_layer - i - 1]]
        blocks.append(
            dgl.create_block(('csc', (ptr, idx, torch.tensor([]))),
                             int(num_src), int(num_dst)))
    return blocks


@hydra.main(version_base=None,
            config_path="../configs/dl",
            config_name="config")
def main(config: DictConfig):
    s = OmegaConf.to_yaml(config)
    log.info(s)
    new_file_name = "new_config.yaml"
    with open(new_file_name, 'w') as f:
        s = s.replace("-", "  -")
        f.write(s)
    loader = cxgnndl.get_loader(config)
    device = torch.device(config.device)
    for iter in range(2):
        torch.cuda.synchronize()
        t0 = time.time()
        for batch in tqdm(loader.train_loader):
            # to_dgl_block(batch)
            # print(batch)
            # print(batch[2][0].type)
            if config.type == "dgl" and config.loading.feat_mode == "uvm":
                load_subtensor_uvm(loader.feat,
                                   loader.graph.ndata['labels'],
                                   batch[1],
                                   batch[0],
                                   device=device)
            # print(batch)
            # print(batch.num_node_in_layer)
            # dump(batch)
            # exit()
            continue
        torch.cuda.synchronize()
        t1 = time.time()
        log.info(f"Time to load batches for an epoch: {t1-t0:.2f} seconds")


if __name__ == '__main__':
    main()
