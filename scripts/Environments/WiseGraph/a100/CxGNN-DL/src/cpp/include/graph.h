#ifndef GRAPH_H
#define GRAPH_H

#include <memory>

#include "Yaml.hpp"
#include "common.h"
#include "file_op.h"
#include "subgraph_index.h"
using std::make_shared;
using std::shared_ptr;
using std::string;

class Graph {
 public:
  SubgraphIndex subgraph_index;
  // coo var
  std::vector<Index> edge_idx;
  // csr var
  std::vector<Index> csr_ptr, csr_idx, num_node_in_layer, num_edge_in_layer, num_etype_in_layer;
  // csr target: in csr format, [ptr[i]:ptr[i + 1]) represents the neighbor of
  // node i if has csr_target, it represents the neighbor of csr_target[i]
  std::vector<Index> csr_target;
  // dataset: label
  std::vector<Index> label;

  std::vector<EtypeIndex> edge_type;

  Yaml::Node config;

  Graph(Yaml::Node &config);

  Graph(std::vector<Index> &&out_edge_idx);

  Graph(std::vector<Index> &&out_csr_ptr, std::vector<Index> &&out_csr_idx,
        std::vector<Index> &&out_num_node_in_layer = std::vector<Index>());

  void toCOO();

  void toCSR();

  inline bool layered() const { return num_node_in_layer.size() >= 1; }
  inline bool hasCOO() const { return coo_avail_; }
  inline bool hasCSR() const { return csr_avail_; }
  inline bool hasTarget() const { return csr_target.size() != 0; }
  inline bool hasLabel() const { return label.size() != 0; }
  inline const Index &getNumNode() const { return num_node_; }
  inline const Index &getNumEdge() const { return num_edge_; }
  inline const int &getNumEtype() const { return num_etype_; }
  inline const Index &getNumUnique() const { return subgraph_index.num_nodes; }
  inline void setNumNode(Index num_node) { num_node_ = num_node; }
  inline void setParentGraph(shared_ptr<Graph> parent_graph) {
    parent_graph_ = parent_graph;
  }
  inline void setTarget(std::vector<Index> &&target) {
    csr_target = std::move(target);
  }
  inline void setNumEdgeInLayer(std::vector<Index> &&out_num_edge_in_layer) {
    num_edge_in_layer = std::move(out_num_edge_in_layer);
  }
  inline void setNumEtypeInLayer(std::vector<Index> &&out_num_etype_in_layer) {
    num_etype_in_layer = std::move(out_num_etype_in_layer);
  }
  inline void setEdgeType(std::vector<EtypeIndex> &&out_edge_type) {
    edge_type = std::move(out_edge_type);
  }
  inline void setLayerInfo(std::vector<Index> &&out_num_node_in_layer,
                           std::vector<Index> &&out_num_edge_in_layer) {
    num_node_in_layer = std::move(out_num_node_in_layer);
    num_edge_in_layer = std::move(out_num_edge_in_layer);
  }
  shared_ptr<Graph> induce(const SubgraphIndex &subgraph_index);
  shared_ptr<Graph> induceLayered(const SubgraphIndex &subgraph_index,
                                  Index num_seed, int num_layer,
                                  const std::vector<Index> &num_node_in_layer);
  void setSubgraphIdx(SubgraphIndex &&subgraph_index) {
    this->subgraph_index = std::move(subgraph_index);
  }
  void setSubgraphIdx(const SubgraphIndex &subgraph_index) {
    this->subgraph_index = SubgraphIndex(subgraph_index);
  }
  void display() {
    SPDLOG_INFO("v {} e {} unique {}", num_node_, num_edge_,
                subgraph_index.num_nodes);
  }
  void toUndirected();
  ~Graph() {}

 protected:
  Index num_node_ = -1;
  Index num_edge_ = -1;
  int num_etype_ = -1;
  bool coo_avail_ = false;
  bool csr_avail_ = false;
  bool is_full_graph_ = false;
  shared_ptr<Graph> parent_graph_ = nullptr;
};

#endif
