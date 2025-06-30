#include "sample.h"
#include "split.h"

SamplerReturnType FullSampler::sample(const shared_ptr<Graph> &graph,
                                      GraphType type) {
  std::vector<Index> mask_in_full = split_->sample(-1);
  if (mask_in_full.size() == 0)
    return std::make_tuple(nullptr, std::vector<Index>(), std::vector<Index>());

  if (type == GraphType::COO) {
    if (!graph->hasCOO()) graph->toCOO();
  } else if (type == GraphType::CSR) {
    if (!graph->hasCSR()) graph->toCSR();
  } else {
    throw std::runtime_error("Unknown graph type");
  }

  return std::make_tuple(graph, mask_in_full, mask_in_full);
}

CpuFullSampler::CpuFullSampler(Yaml::Node &config) {
  batch_size_ = config["batch_size"].As<int>(0);
  ASSERT(batch_size_ > 0 || batch_size_ == -1);
  num_layer_ = config["num_layer"].As<int>(0);
  ASSERT(num_layer_ > 0);
}

inline void expand(const shared_ptr<Graph> &graph,
                   SubgraphIndex &subgraph_index, std::vector<Index> &src,
                   std::vector<Index> &dest, const Index &sub_id) {
  Index full_id = subgraph_index.sub_to_full[sub_id];
  Index begin = graph->csr_ptr[full_id], end = graph->csr_ptr[full_id + 1];
  Index num_neighbor = end - begin;
  int cnt = 0;
  if (num_neighbor == 0) {
    dest.push_back(dest.back() + cnt);
    return;
  }
  for (Index j = begin; j < end; ++j) {
    src.push_back(subgraph_index.addNode(graph->csr_idx[j]));
  }
  cnt += num_neighbor;
  dest.push_back(dest.back() + cnt);
}

SamplerReturnType CpuFullSampler::sample(const shared_ptr<Graph> &graph,
                                         GraphType type) {
  assert(type == GraphType::CSR_Layer);
  std::vector<Index> seed_nodes =
      split_->sample(batch_size_);  // all the nodes in a set
  if (seed_nodes.size() == 0)
    return std::make_tuple(nullptr, std::vector<Index>(), std::vector<Index>());
  SubgraphIndex subgraph_index;

  std::vector<Index> src, dest;
  Index seed_begin = 0;
  Index seed_end = seed_nodes.size();
  std::vector<Index> num_node_in_layer = {seed_end};

  for (Index i = seed_begin; i < seed_end; ++i) {
    subgraph_index.addNode(seed_nodes[i]);
  }
  dest.push_back(0);

  for (int layer_id = num_layer_ - 1; layer_id >= 0; --layer_id) {
    for (Index i = seed_begin; i < seed_end; ++i) {
      expand(graph, subgraph_index, src, dest, i);
    }
    seed_begin = seed_end;
    seed_end = subgraph_index.num_nodes;
    // Other Types (e.g., CSR_Layer, CSR_Schedule do not need to resize at layer
    // 0)
    num_node_in_layer.push_back(seed_end);  // for Layered CSR format
  }
  std::vector<Index> mask_in_sub(seed_nodes.size());
  std::iota(mask_in_sub.begin(), mask_in_sub.end(), 0);
  shared_ptr<Graph> sampled_graph = nullptr;
  SPDLOG_WARN("subgraph {} {}", dest.size(), src.size());
  sampled_graph =
      make_shared<Graph>(std::move(dest) /*ptr*/, std::move(src) /*idx*/,
                         std::move(num_node_in_layer));
  SPDLOG_WARN("subgraph2 {} {}", sampled_graph->csr_ptr.size(),
              sampled_graph->csr_idx.size());
  sampled_graph->setSubgraphIdx(std::move(subgraph_index));
  sampled_graph->setParentGraph(graph);
  return std::make_tuple(sampled_graph, std::move(seed_nodes),
                         std::move(mask_in_sub));
}