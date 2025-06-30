#include "sample.h"
#include "split.h"

CpuKHopSampler::CpuKHopSampler(Yaml::Node &config) {
  Yaml::Node &fanout = config["fanouts"];
  ASSERTWITH(fanout.IsSequence(), "fanouts");
  int cnt = 0;
  for (auto it = fanout.Begin(); it != fanout.End(); it++) {
    fanouts_.push_back((*it).second.As<int>());
    SPDLOG_INFO("fanout[{}]: {}", cnt, fanouts_.back());
    cnt++;
    if (cnt > 10) throw std::runtime_error("Too many layer fanouts");
  }
  replace = config["replace"].As<bool>(false);
  batch_size_ = config["batch_size"].As<int>(0);
  SPDLOG_INFO(" batch size {}", batch_size_);
  ASSERT(fanouts_.size() > 0);
  ASSERT(batch_size_ != 0);
}

inline void expand(const shared_ptr<Graph> &graph,
                   SubgraphIndex &subgraph_index, uint32_t &random_seed,
                   const Index &sub_id, const int &fanout, bool replace) {
  Index full_id = subgraph_index.sub_to_full[sub_id];
  Index begin = graph->csr_ptr[full_id], end = graph->csr_ptr[full_id + 1];
  Index num_neighbor = end - begin;
  if (num_neighbor == 0) return;

  int cnt = 0;
  // number of neighbors less than fanout, get all neighbors
  if (fanout < 0 || (!replace && num_neighbor <= fanout)) {
    for (Index j = begin; j < end; ++j)
      subgraph_index.addNode(graph->csr_idx[j]);
    cnt = num_neighbor;
  } else if (replace) {
    // sample with replacement
    for (int k = 0; k < fanout; ++k) {
      int tmp = randInt(0, num_neighbor, random_seed);
      subgraph_index.addNode(graph->csr_idx[tmp + begin]);
    }
    cnt = fanout;
  } else {
    // Donald Knuth's sample without replacement algorithm
    for (Index k = 0; cnt < fanout; ++k) {
      // call a uniform(0,1) random number generator
      double p = zeroToOne(random_seed);
      // k has (fanout - cnt) / (num_neighbor - k) chance of being selected
      if ((num_neighbor - k) * p < fanout - cnt) {
        subgraph_index.addNode(graph->csr_idx[k + begin]);
        ++cnt;
      }
    }
  }
}

SamplerReturnType CpuKHopSampler::sample(const shared_ptr<Graph> &graph,
                                         GraphType type) {
  std::vector<Index> seed_nodes = split_->sample(batch_size_);
  if (seed_nodes.size() == 0)
    return std::make_tuple(nullptr, std::vector<Index>(), std::vector<Index>());
  SubgraphIndex subgraph_index;

  Index seed_begin = 0;
  Index seed_end = seed_nodes.size();
  uint32_t random_seed = time(NULL);

  for (Index i = 0; i < seed_nodes.size(); ++i)
    subgraph_index.addNode(seed_nodes[i]);

  for (int layer_id = fanouts_.size() - 1; layer_id >= 0; --layer_id) {
    int fanout = fanouts_[layer_id];
    for (Index i = seed_begin; i < seed_end; ++i)
      expand(graph, subgraph_index, random_seed, i, fanout, replace);
    seed_begin = seed_end;
    seed_end = subgraph_index.num_nodes;
  }
  return postSample(seed_nodes, subgraph_index, graph, type);
}

SamplerReturnType CpuKHopSampler::postSample(
    vector<Index> &seed_nodes, const SubgraphIndex &subgraph_index,
    const shared_ptr<Graph> &graph, GraphType type) {
  shared_ptr<Graph> sampled_graph = graph->induce(subgraph_index);
  if (type == GraphType::COO)
    sampled_graph->toCOO();
  else if (type == GraphType::CSR)
    ;  // do nothing
  else
    throw std::runtime_error("Unsupported graph type");

  sampled_graph->setParentGraph(graph);
  std::vector<Index> mask_in_sub(seed_nodes.size());
  std::iota(mask_in_sub.begin(), mask_in_sub.end(), 0);
  return std::make_tuple(sampled_graph, std::move(seed_nodes),
                         std::move(mask_in_sub));
}
