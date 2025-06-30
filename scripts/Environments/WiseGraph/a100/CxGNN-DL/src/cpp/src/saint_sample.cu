#include "sample.h"
#include "split.h"

SaintSampler::SaintSampler(Yaml::Node &config) {
  Yaml::Node &fanout = config["fanouts"];
  ASSERTWITH(fanout.IsSequence(), "fanouts");
  int cnt = 0;
  for (auto it = fanout.Begin(); it != fanout.End(); it++) {
    fanouts_.push_back((*it).second.As<int>());
    // SPDLOG_INFO("fanout[{}]: {}", cnt, fanouts_.back());
    cnt++;
    if (cnt > 10) throw std::runtime_error("Too many layer fanouts");
  }
  batch_size_ = config["batch_size"].As<int>(0);
  num_layers_ = config["num_layer"].As<int>(0);
  ASSERT(batch_size_ != 0);
}

SamplerReturnType SaintSampler::sample(const shared_ptr<Graph> &graph,
                                       GraphType type) {
  std::vector<Index> seed_nodes = split_->sample(batch_size_);
  if (seed_nodes.size() == 0)
    return std::make_tuple(nullptr, std::vector<Index>(), std::vector<Index>());
  ASSERTWITH(type == GraphType::CSR_Layer,
             "SaintSampler only supports CSR_Layer graph");
  SubgraphIndex subgraph_index;
  // pre-sample
  for (Index i = 0; i < seed_nodes.size(); ++i) {
    subgraph_index.addNode(seed_nodes[i]);
  }
  uint32_t random_seed = time(NULL);
  Index seed_begin = 0, seed_end = seed_nodes.size();
  for (int layer_id = fanouts_.size() - 1; layer_id >= 0; --layer_id) {
    int fanout = fanouts_[layer_id];
    for (int i = seed_begin; i < seed_end; ++i) {
      // int tmp_v = randInt(0, num_node_all, random_seed);
      Index center = subgraph_index.sub_to_full[i];
      for (int j = 0; j < fanout; ++j) {
        int tmp_e =
            randInt(0, graph->csr_ptr[center + 1] - graph->csr_ptr[center],
                    random_seed);
        subgraph_index.addNode(graph->csr_idx[graph->csr_ptr[center] + tmp_e]);
      }
    }
    seed_begin = seed_end;
    seed_end = subgraph_index.sub_to_full.size();
  }

  auto ret_graph = graph->induce(subgraph_index);
  Index num_node = ret_graph->getNumNode();
  Index num_edge = ret_graph->getNumEdge();
  ASSERT(num_node != -1 && num_edge != -1);
  std::vector<Index> num_node_in_layer;
  std::vector<Index> num_edge_in_layer;
  for (Index i = 0; i < num_layers_; ++i) {
    num_node_in_layer.push_back(num_node);
    num_edge_in_layer.push_back(num_edge);
  }
  std::vector<Index> mask_in_sub(subgraph_index.sub_to_full.size());
  std::iota(mask_in_sub.begin(), mask_in_sub.end(), 0);

  ret_graph->setLayerInfo(std::move(num_node_in_layer),
                          std::move(num_edge_in_layer));
  return std::make_tuple(ret_graph, std::move(subgraph_index.sub_to_full),
                         std::move(mask_in_sub));
}