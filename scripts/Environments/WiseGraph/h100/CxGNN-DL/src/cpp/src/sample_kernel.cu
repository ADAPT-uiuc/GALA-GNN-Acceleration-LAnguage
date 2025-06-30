#include "sample_kernel.h"

Tensor vec2tensor(const std::vector<Index> &vec) {
  Tensor t = torch::empty({(int64_t)vec.size()}, torch::kInt64);
  memcpy(t.data_ptr(), vec.data(), vec.size() * sizeof(Index));
  return t;
}

inline void expand(Index *ptr, Index *idx, SubgraphIndex &subgraph_index,
                   std::vector<Index> &src, std::vector<Index> &dest,
                   Index &head_src, Index &head_dest, uint32_t &random_seed,
                   const Index &sub_id, const int &fanout) {
  Index full_id = subgraph_index.sub_to_full[sub_id];
  Index begin = ptr[full_id], end = ptr[full_id + 1];
  Index num_neighbor = end - begin;
  bool replace = false;
  int cnt = 0;
  if (num_neighbor == 0) {
    dest[sub_id + 1] = dest[sub_id] + cnt;
    return;
  }
  // number of neighbors less than fanout, get all neighbors
  if (fanout < 0 || (!replace && num_neighbor <= fanout)) {
    for (Index j = begin; j < end; ++j) {
      src[head_src++] = subgraph_index.addNode(idx[j]);
    }
    cnt += num_neighbor;
  } else if (replace) {
    // sample with replacement
    for (int k = 0; k < fanout; ++k) {
      int tmp = randInt(0, num_neighbor, random_seed);
      src[head_src++] = subgraph_index.addNode(idx[tmp + begin]);
    }
    cnt += fanout;
  } else {
    // Donald Knuth's sample without replacement algorithm
    for (Index k = 0; cnt < fanout; ++k) {
      // call a uniform(0,1) random number generator
      double p = zeroToOne(random_seed);
      // k has (fanout - cnt) / (num_neighbor - k) chance of being selected
      if ((num_neighbor - k) * p < fanout - cnt) {
        src[head_src++] = subgraph_index.addNode(idx[k + begin]);
        ++cnt;
      }
    }
  }
  dest[sub_id + 1] = dest[sub_id] + cnt;
}

std::vector<Tensor> neighbor_sample(Tensor ptr, Tensor idx,
                                    const std::vector<int> &fanouts,
                                    Tensor seed_nodes) {
  SubgraphIndex subgraph_index;
  std::vector<Index> src, dest;
  Index seed_begin = 0;
  Index seed_end = seed_nodes.sizes()[0];
  Index *p_seed_nodes = seed_nodes.data_ptr<Index>();
  std::vector<Index> num_node_in_layer = {seed_end};
  std::vector<Index> num_edge_in_layer;
  uint32_t random_seed = time(NULL);

  for (Index i = 0; i < seed_end; ++i) {
    subgraph_index.addNode(p_seed_nodes[i]);
  }
  dest.resize(seed_end + 1);
  dest[0] = 0;
  Index *p_ptr = ptr.data<Index>();
  Index *p_idx = idx.data<Index>();

  for (int layer_id = fanouts.size() - 1; layer_id >= 0; --layer_id) {
    int fanout = fanouts[layer_id];
    Index head_src = src.size();
    Index head_dest = dest.size();
    src.resize(src.size() + (seed_end - seed_begin) * fanout);
    for (Index i = seed_begin; i < seed_end; ++i) {
      expand(p_ptr, p_idx, subgraph_index, src, dest, head_src, head_dest,
             random_seed, i, fanout);
    }
    seed_begin = seed_end;
    seed_end = subgraph_index.num_nodes;
    src.resize(head_src);
    num_node_in_layer.push_back(seed_end);  // for Layered CSR format
    num_edge_in_layer.push_back(head_src);
    // Other Types (e.g., CSR_Layer, CSR_Schedule do not need to resize at
    // layer 0)
    if (layer_id != 0) {
      dest.resize(subgraph_index.num_nodes + 1);
    }
  }
  std::vector<Tensor> ret;
  ret.push_back(vec2tensor(dest));
  ret.push_back(vec2tensor(src));
  ret.push_back(vec2tensor(subgraph_index.sub_to_full));
  ret.push_back(vec2tensor(num_node_in_layer));
  ret.push_back(vec2tensor(num_edge_in_layer));
  return ret;
}