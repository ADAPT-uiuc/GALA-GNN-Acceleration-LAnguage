#include "sample.h"
#include "split.h"

CpuNeighborTypeSampler::CpuNeighborTypeSampler(Yaml::Node &config)
    : CpuNeighborSampler(config) {}

SamplerReturnType CpuNeighborTypeSampler::postSample(
    vector<Index> &src, vector<Index> &dest, vector<EtypeIndex> &edge_type,
    vector<Index> &seed_nodes, SubgraphIndex &subgraph_index,
    vector<Index> num_node_in_layer, vector<Index> num_edge_in_layer, vector<Index> num_etype_in_layer,
    const shared_ptr<Graph> &graph, GraphType type) {
  std::vector<Index> mask_in_sub(seed_nodes.size());
  std::iota(mask_in_sub.begin(), mask_in_sub.end(), 0);
  shared_ptr<Graph> sampled_graph = nullptr;
  if (type == GraphType::COO) {
    // generate edge_index
    src.insert(src.end(), std::make_move_iterator(dest.begin()),
               std::make_move_iterator(dest.end()));
    sampled_graph = make_shared<Graph>(std::move(src));
    sampled_graph->setNumNode(subgraph_index.num_nodes);
    sampled_graph->setLayerInfo(std::move(num_node_in_layer),
                                std::move(num_edge_in_layer));
  } else {
    sampled_graph =
        make_shared<Graph>(std::move(dest) /*ptr*/, std::move(src) /*idx*/,
                           std::move(num_node_in_layer));
    sampled_graph->setNumEdgeInLayer(std::move(num_edge_in_layer));
  }
  if (type == GraphType::CSR_Layer || type == GraphType::CSR_Schedule) {
    ASSERT(sampled_graph->layered())
  }
  sampled_graph->setSubgraphIdx(std::move(subgraph_index));
  sampled_graph->setParentGraph(graph);
  sampled_graph->setEdgeType(std::move(edge_type));
  sampled_graph->setNumEtypeInLayer(std::move(num_etype_in_layer));
  return std::make_tuple(sampled_graph, std::move(seed_nodes),
                         std::move(mask_in_sub));
}

inline void expand(const shared_ptr<Graph> &graph,
                   SubgraphIndex &subgraph_index, std::vector<Index> &src,
                   std::vector<Index> &dest, std::vector<EtypeIndex> &etype,
                   Index *num_edge_per_rel,
                   Index &head_src, Index &head_dest, uint32_t &random_seed,
                   const Index &sub_id, const int &fanout,
                   const GraphType &type, const bool &replace,
                   const bool &self_loop) {
  Index full_id = subgraph_index.sub_to_full[sub_id];
  Index begin = graph->csr_ptr[full_id], end = graph->csr_ptr[full_id + 1];
  Index num_neighbor = end - begin;
  // SPDLOG_INFO("{} {}", num_neighbor, fanout);
  int cnt = 0;
  if (self_loop) {
    src[head_src] = sub_id;
    etype[head_src] = 0;  // self loop type set to 0
    num_edge_per_rel[0]++;
    ++head_src;
    cnt = 1;
  }
  if (num_neighbor == 0) {
    if (type != GraphType::COO)
      dest[sub_id + 1] =
          dest[sub_id] + cnt;  // BUG fixed, CSR requires prefix summation
    else if (cnt == 1) {
      // COO and self loop
      dest[head_dest++] = sub_id;
    }
    return;
  }
  // number of neighbors less than fanout, get all neighbors
  if (fanout < 0 || (!replace && num_neighbor <= fanout)) {
    for (Index j = begin; j < end; ++j) {
      src[head_src] = subgraph_index.addNode(graph->csr_idx[j]);
      etype[head_src] = graph->edge_type[j];
      num_edge_per_rel[graph->edge_type[j]]++;
      ++head_src;
    }
    cnt += num_neighbor;
  } else if (replace) {
    // sample with replacement
    for (int k = 0; k < fanout; ++k) {
      int tmp = randInt(0, num_neighbor, random_seed);
      src[head_src] = subgraph_index.addNode(graph->csr_idx[tmp + begin]);
      etype[head_src] = graph->edge_type[tmp + begin];
      num_edge_per_rel[graph->edge_type[tmp + begin]]++;
      ++head_src;
    }
    cnt += fanout;
  } else {
    // Donald Knuth's sample without replacement algorithm
    for (Index k = 0; cnt < fanout; ++k) {
      // call a uniform(0,1) random number generator
      double p = zeroToOne(random_seed);
      // k has (fanout - cnt) / (num_neighbor - k) chance of being selected
      if ((num_neighbor - k) * p < fanout - cnt) {
        if (!self_loop || graph->csr_idx[k + begin] != full_id) {
          src[head_src] = subgraph_index.addNode(graph->csr_idx[k + begin]);
          etype[head_src] = graph->edge_type[k + begin];
          num_edge_per_rel[graph->edge_type[k + begin]]++;
          ++head_src;
          ++cnt;
        }
      }
    }
  }
  if (type == GraphType::COO) {
    for (int _ = 0; _ < cnt; ++_) dest[head_dest++] = sub_id;
  } else {
    dest[sub_id + 1] = dest[sub_id] + cnt;
  }
}

SamplerReturnType CpuNeighborTypeSampler::sample(const shared_ptr<Graph> &graph,
                                                 GraphType type) {
  std::vector<Index> seed_nodes = split_->sample(batch_size_);
  // return empty subgraph
  if (seed_nodes.size() == 0)
    return std::make_tuple(nullptr, std::vector<Index>(), std::vector<Index>());

  SubgraphIndex subgraph_index;
  std::vector<Index> src, dest;
  std::vector<EtypeIndex> etype;
  Index seed_begin = 0;
  Index seed_end = seed_nodes.size();
  std::vector<Index> num_node_in_layer = {seed_end};
  std::vector<Index> num_edge_in_layer;
  int num_etype = graph->getNumEtype();
  std::vector<Index> num_edge_per_rel(num_etype * fanouts_.size(), 0);
  uint32_t random_seed = time(NULL);

  preSample(src, dest, subgraph_index, seed_nodes, type);

  for (int layer_id = fanouts_.size() - 1; layer_id >= 0; --layer_id) {
    int fanout = fanouts_[layer_id];
    Index head_src = src.size();
    Index head_dest = dest.size();
    ASSERTWITH(type != GraphType::COO || fanout > 0,
               "Cannot use negative fanout, the following code uses fanout to "
               "pre-allocate the buffer");
    src.resize(src.size() + (seed_end - seed_begin) * fanout);
    etype.resize(etype.size() + (seed_end - seed_begin) * fanout);
    if (type == GraphType::COO)
      dest.resize(dest.size() + (seed_end - seed_begin) * fanout);
    // timestamp(t0);
    for (Index i = seed_begin; i < seed_end; ++i) {
      expand(graph, subgraph_index, src, dest, etype, num_edge_per_rel.data() + layer_id * num_etype, head_src, head_dest,
             random_seed, i, fanout, type, replace, self_loop);
    }
    seed_begin = seed_end;
    seed_end = subgraph_index.num_nodes;
    src.resize(head_src);
    etype.resize(head_src);
    num_node_in_layer.push_back(seed_end);  // for Layered CSR format
    num_edge_in_layer.push_back(head_src);
    if (type != GraphType::COO) {
      // Other Types (e.g., CSR_Layer, CSR_Schedule do not need to resize at
      // layer 0)
      if (type == GraphType::CSR || layer_id != 0) {
        dest.resize(subgraph_index.num_nodes + 1);
      }
    } else if (type == GraphType::COO){
      dest.resize(head_dest);
    }
    if (layer_id != 0) {
      for (int _ = 0; _ < num_etype; ++_) {
        num_edge_per_rel[(layer_id - 1) * num_etype + _] = num_edge_per_rel[layer_id * num_etype + _];
      }
    }
  }
  if (type == GraphType::CSR) {
    for (Index i = seed_begin; i < seed_end; ++i) dest[i + 1] = dest[i];
  }

  ASSERT(src.size() == etype.size());
  return postSample(src, dest, etype, seed_nodes, subgraph_index,
                    num_node_in_layer, num_edge_in_layer, num_edge_per_rel, graph, type);
}