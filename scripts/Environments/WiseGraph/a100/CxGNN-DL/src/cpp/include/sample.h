#ifndef SAMPLE_H
#define SAMPLE_H

#include <stdint.h>

#include <memory>
#include <tuple>
#include <vector>

#include "Yaml.hpp"
#include "common.h"
#include "graph.h"
#include "phmap.h"

class Split;
using phmap::flat_hash_map;
using phmap::flat_hash_set;
using std::vector;
// using phmap::parallel_flat_hash_map;

enum class GraphType { COO, CSR, CSR_Layer, CSR_Schedule };

using std::make_shared;
using std::shared_ptr;

// (subgraph, mask_in_full, mask_in_sub)
using SamplerReturnType =
    std::tuple<shared_ptr<Graph>, vector<Index>, vector<Index>>;

class Sampler {
 public:
  static shared_ptr<Sampler> create(Yaml::Node &config);
  virtual SamplerReturnType sample(const shared_ptr<Graph> &graph,
                                   GraphType type = GraphType::COO) = 0;
  virtual void set_split(shared_ptr<Split> split);

 protected:
  shared_ptr<Split> split_;
};
class FullSampler : public Sampler {
 public:
  FullSampler() {}

  virtual SamplerReturnType sample(const shared_ptr<Graph> &graph,
                                   GraphType type = GraphType::COO) override;
};

class CpuNeighborSampler : public Sampler {
 public:
  CpuNeighborSampler(Yaml::Node &config);

  virtual SamplerReturnType sample(const shared_ptr<Graph> &graph,
                                   GraphType type = GraphType::COO) override;

 protected:
  vector<int> fanouts_;
  Index batch_size_;
  bool replace = false;
  bool self_loop = false;

  void preSample(vector<Index> &src, vector<Index> &dest,
                 SubgraphIndex &subgraph_index, const vector<Index> &seed_nodes,
                 GraphType type);

  SamplerReturnType postSample(vector<Index> &src, vector<Index> &dest,
                               vector<Index> &seed_nodes,
                               SubgraphIndex &subgraph_index,
                               vector<Index> num_node_in_layer,
                               vector<Index> num_edge_in_layer,
                               const shared_ptr<Graph> &graph, GraphType type);
};

class CpuNeighborTypeSampler : public CpuNeighborSampler {
 public:
  CpuNeighborTypeSampler(Yaml::Node &config);

  virtual SamplerReturnType sample(const shared_ptr<Graph> &graph,
                                   GraphType type = GraphType::COO) override;

 protected:
  SamplerReturnType postSample(vector<Index> &src, vector<Index> &dest,
                               vector<EtypeIndex> &edge_type,
                               vector<Index> &seed_nodes,
                               SubgraphIndex &subgraph_index,
                               vector<Index> num_node_in_layer,
                               vector<Index> num_edge_in_layer,
                               vector<Index> num_etype_in_layer,
                               const shared_ptr<Graph> &graph, GraphType type);
};

class CpuFullSampler : public Sampler {
 public:
  CpuFullSampler(Yaml::Node &config);

  virtual SamplerReturnType sample(const shared_ptr<Graph> &graph,
                                   GraphType type = GraphType::COO) override;

 protected:
  Index batch_size_ = -1;
  int num_layer_ = -1;
};

class CpuKHopSampler : public Sampler {
 public:
  CpuKHopSampler(Yaml::Node &config);

  virtual SamplerReturnType sample(const shared_ptr<Graph> &graph,
                                   GraphType type = GraphType::COO) override;

 protected:
  vector<int> fanouts_;
  Index batch_size_;
  bool replace = false;

  SamplerReturnType postSample(vector<Index> &seed_nodes,
                               const SubgraphIndex &subgraph_index,
                               const shared_ptr<Graph> &graph, GraphType type);
};

enum class ClusterAggrType { ClusterGCN, GAS, GraphFM };
class ClusterSampler : public Sampler {
 public:
  ClusterSampler(Yaml::Node &config);

  virtual SamplerReturnType sample(const shared_ptr<Graph> &graph,
                                   GraphType type = GraphType::COO) override;

 protected:
  Index batch_size_;  // number of clusters per batch
  int num_layers_;
  ClusterAggrType cluster_mode = ClusterAggrType::ClusterGCN;
};

class SaintSampler : public Sampler {
 public:
  SaintSampler(Yaml::Node &config);

  virtual SamplerReturnType sample(const shared_ptr<Graph> &graph,
                                   GraphType type = GraphType::COO) override;

 protected:
  Index batch_size_ = 0;  // number of edges
  int num_layers_ = 0;
  vector<int> fanouts_;
};

#endif
