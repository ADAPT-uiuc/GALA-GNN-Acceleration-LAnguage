#ifndef __SPLIT_H__
#define __SPLIT_H__

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "Yaml.hpp"
#include "common.h"
class Split {
  /*
   * A split is a set of nodes in the graph.
   * split_idx: the index of the nodes in the split
   * num_split_node: the number of nodes in the split
   * split: the name of the split
   */
 public:
  std::vector<Index> split_idx;
  std::vector<bool> target_set_mask;
  std::vector<Index> cluster_ptr;
  std::vector<Index> cluster_idx;
  static std::vector<Index> cluster;
  Index num_split_node = 0;
  std::string split;
  Split(bool shuffle = true) : head(0), shuffle(shuffle) {}
  Split(std::vector<Index> &&split_idx, bool shuffle = true);
  Split(std::vector<Index> &&split_idx, Index num_split_node, std::string split,
        bool shuffle = true)
      : split_idx(std::move(split_idx)),
        num_split_node(num_split_node),
        split(split),
        head(0),
        shuffle(shuffle) {}

  Split(Yaml::Node &config, std::string split, int upper_limit = -1,
        bool shuffle = true);

  inline Index operator[](const Index &idx) const { return split_idx[idx]; }
  int fetchAdd(int size) {
    int ret = head.fetch_add(size);
    if (ret >= split_idx.size()) return -1;
    return ret;
  }

  std::vector<Index> sample(Index num_samples) {
    std::vector<Index> sequence;
    int ret = 0;
    if (num_samples == -1) {  // fetch all nodes
      num_samples = num_split_node;
      ret = fetchAdd(num_samples);
      if (ret != 0) return sequence;
    } else {
      ret = fetchAdd(num_samples);
      if (ret == -1) return sequence;
      if (ret + num_samples > this->size()) num_samples = this->size() - ret;
    }
    sequence.resize(num_samples);
    memcpy(sequence.data(), split_idx.data() + ret,
           num_samples * sizeof(Index));
    return sequence;
  }

  std::vector<Index> sample_cluster(Index num_samples);

  void setRepeat(int out_repeat);

  int size() const { return split_idx.size(); }
  void reset();

  bool shuffle = true;
  uint32_t random_seed = 233;
  std::mt19937 random_engine = std::mt19937(random_seed);
  std::atomic<int> head;
  int upper_limit = -1;
  int repeat = 1;
};
#endif  // __SPLIT_H__