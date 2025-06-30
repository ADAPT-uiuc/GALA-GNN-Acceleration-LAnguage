#include "sample.h"
#include "split.h"

ClusterSampler::ClusterSampler(Yaml::Node &config) {
  batch_size_ = config["batch_size"].As<int>(0);
  num_layers_ = config["num_layer"].As<int>(0);
  string cluster_name = config["cluster_mode"].As<string>();
  if (cluster_name == "clustergcn")
    cluster_mode = ClusterAggrType::ClusterGCN;
  else if (cluster_name == "gas")
    cluster_mode = ClusterAggrType::GAS;
  else if (cluster_name == "graphfm")
    cluster_mode = ClusterAggrType::GraphFM;
  else {
    ASSERTWITH(false, "the cluster mode is invalid {}", cluster_name);
  }
  ASSERT(batch_size_ > 0);
  ASSERT(num_layers_ > 0);
}

SamplerReturnType ClusterSampler::sample(const shared_ptr<Graph> &graph,
                                         GraphType type) {
  ASSERT(GraphType::COO != type);
  ASSERT(split_->target_set_mask.size() > 0);
  std::vector<Index> seed_nodes = split_->sample_cluster(batch_size_);
  if (seed_nodes.size() == 0)
    return std::make_tuple(nullptr, std::vector<Index>(), std::vector<Index>());
  SubgraphIndex subgraph_index;
  for (Index i = 0; i < seed_nodes.size(); ++i) {
    subgraph_index.addNode(seed_nodes[i]);
  }
  std::vector<Index> mask_in_sub;
  std::vector<Index> mask_in_full;
  for (Index i = 0; i < seed_nodes.size(); ++i) {
    if (split_->target_set_mask[seed_nodes[i]]) {
      mask_in_sub.push_back(i);
      mask_in_full.push_back(seed_nodes[i]);
    }
  }
  Index num_seed = seed_nodes.size();
  std::shared_ptr<Graph> ret_graph = nullptr;
  if (cluster_mode == ClusterAggrType::GAS) {
    std::vector<Index> src, dest;
    dest.push_back(0);
    for (Index i = 0; i < seed_nodes.size(); ++i) {
      Index full_id = seed_nodes[i];
      Index begin = graph->csr_ptr[full_id], end = graph->csr_ptr[full_id + 1];
      Index cnt = 0;
      for (Index j = begin; j < end; ++j) {
        src.push_back(subgraph_index.addNode(graph->csr_idx[j]));
      }
      cnt += end - begin;
      dest.push_back(dest.back() + cnt);
    }
    std::vector<Index> num_node_in_layer;
    std::vector<Index> num_edge_in_layer;
    Index num_node = seed_nodes.size();
    Index num_edge = src.size();
    for (Index i = 0; i < num_layers_ + 1; ++i) {
      num_node_in_layer.push_back(num_node);
      num_edge_in_layer.push_back(num_edge);
    }
    ret_graph =
        make_shared<Graph>(std::move(dest) /*ptr*/, std::move(src) /*idx*/,
                           std::move(num_node_in_layer));
    ret_graph->setNumEdgeInLayer(std::move(num_edge_in_layer));
    ret_graph->setSubgraphIdx(subgraph_index);
    ret_graph->setParentGraph(graph);
  } else if (cluster_mode == ClusterAggrType::ClusterGCN) {
    ret_graph = graph->induce(subgraph_index);
    Index num_node = ret_graph->getNumNode();
    Index num_edge = ret_graph->getNumEdge();
    ASSERT(num_node == ret_graph->csr_ptr.size() - 1);
    ASSERT(num_edge == ret_graph->csr_idx.size());
    std::vector<Index> num_node_in_layer;
    std::vector<Index> num_edge_in_layer;
    for (Index i = 0; i < num_layers_ + 1; ++i) {
      num_node_in_layer.push_back(num_node);
      num_edge_in_layer.push_back(num_edge);
    }
    SPDLOG_INFO("Sampled {} nodes and {} edges {} seeds", num_node, num_edge,
                seed_nodes.size());
    ret_graph->setLayerInfo(std::move(num_node_in_layer),
                            std::move(num_edge_in_layer));
  } else if (cluster_mode == ClusterAggrType::GraphFM) {
    for (Index i = 0; i < seed_nodes.size(); ++i) {
      Index full_id = seed_nodes[i];
      Index begin = graph->csr_ptr[full_id], end = graph->csr_ptr[full_id + 1];
      for (Index j = begin; j < end; ++j) {
        subgraph_index.addNode(graph->csr_idx[j]);
      }
    }
    ret_graph = graph->induce(subgraph_index);
    Index num_node = ret_graph->getNumNode();
    Index num_edge = ret_graph->getNumEdge();
    std::vector<Index> num_node_in_layer;
    std::vector<Index> num_edge_in_layer;
    for (Index i = 0; i < num_layers_; ++i) {
      num_node_in_layer.push_back(num_seed);
      num_edge_in_layer.push_back(num_edge);
    }
    num_node_in_layer.push_back(num_node);
    num_edge_in_layer.push_back(num_edge);
    ret_graph->setLayerInfo(std::move(num_node_in_layer),
                            std::move(num_edge_in_layer));
  }
  return std::make_tuple(ret_graph, std::move(mask_in_full),
                         std::move(mask_in_sub));
}