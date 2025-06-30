#ifndef COMPUTE_SCHEDULE_H
#define COMPUTE_SCHEDULE_H
#include <memory>
#include <vector>
#include "common.h"
using std::shared_ptr;
using std::vector;

#include "common.h"
#include "graph.h"

class ComputeScheduler {
 public:
  static shared_ptr<ComputeScheduler> create(Yaml::Node &config) {
    return make_shared<ComputeScheduler>(config);
  }

  ComputeScheduler(Yaml::Node &config);
  vector<shared_ptr<Graph>> schedule(shared_ptr<Graph> graph, int layer_id,
                                     torch::Tensor cache = torch::Tensor());
  // break the graph into two parts
  vector<shared_ptr<Graph>> breakGraph(shared_ptr<Graph> graph,
                                       const vector<Index> &nodes);

 private:
  //   XGB *model;
  int iter_limit = 0;
  int degree_thres = 0;
  int feat_in = 0;
  float score_thres = 0;
  float graph2mm = 0;
};

std::vector<torch::Tensor> rel_schedule(shared_ptr<Graph> subgraph,
                                        const std::vector<EtypeIndex> &rel);
#endif