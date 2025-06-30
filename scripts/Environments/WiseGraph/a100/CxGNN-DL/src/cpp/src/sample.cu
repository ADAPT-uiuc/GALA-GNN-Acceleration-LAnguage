#include <stdlib.h>

#include <algorithm>
#include <random>
#include <sstream>
#include <vector>

#include "common.h"
#include "dataset.h"
#include "graph.h"
#include "sample.h"
#include "spdlog/spdlog.h"
#include "split.h"
#include "timer.h"

shared_ptr<Sampler> Sampler::create(Yaml::Node &config) {
  std::string sampler = config["name"].As<string>();
  if (sampler == "neighbor")
    return make_shared<CpuNeighborSampler>(config);
  else if (sampler == "khop")
    return make_shared<CpuKHopSampler>(config);
  else if (sampler == "full")
    return make_shared<FullSampler>();
  else if (sampler == "full_layer")
    return make_shared<CpuFullSampler>(config);
  else if (sampler == "typed_neighbor")
    return make_shared<CpuNeighborTypeSampler>(config);
  else if (sampler == "cluster")
    return make_shared<ClusterSampler>(config);
  else if (sampler == "saint")
    return make_shared<SaintSampler>(config);
  else
    throw std::runtime_error("Unknown sampler: " + sampler);
}

void Sampler::set_split(std::shared_ptr<Split> split) { split_ = split; }
