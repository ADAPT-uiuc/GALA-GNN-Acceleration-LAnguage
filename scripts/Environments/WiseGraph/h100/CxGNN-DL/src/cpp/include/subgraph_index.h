#ifndef SUBGRAPH_INDEX_H_
#define SUBGRAPH_INDEX_H_
#include "common.h"
#include "phmap.h"

using phmap::flat_hash_map;
using phmap::flat_hash_set;

struct SubgraphIndex {
  inline Index addNode(Index node) {
    auto it = full_to_sub.find(node);
    if (it == full_to_sub.end()) {
      sub_to_full.push_back(node);
      full_to_sub[node] = num_nodes;
      return num_nodes++;
    }
    return it->second;
  }
  inline Index findNode(Index node) {
    auto it = full_to_sub.find(node);
    if (it == full_to_sub.end()) {
      return -1;
    }
    return it->second;
  }
  SubgraphIndex(Index num_nodes = 0, bool is_full = false)
      : num_nodes(num_nodes), is_full(is_full) {}
  SubgraphIndex(std::vector<Index> &&sub_to_full,
                flat_hash_map<Index, Index> &&full_to_sub, Index num_nodes)
      : sub_to_full(std::move(sub_to_full)),
        full_to_sub(std::move(full_to_sub)),
        num_nodes(num_nodes) {}
  SubgraphIndex(std::vector<Index> &&sub_to_full,
                flat_hash_map<Index, Index> &&full_to_sub)
      : sub_to_full(std::move(sub_to_full)),
        full_to_sub(std::move(full_to_sub)),
        num_nodes(sub_to_full.size()) {}
  std::vector<Index> sub_to_full;
  flat_hash_map<Index, Index> full_to_sub;
  Index num_nodes = 0;
  bool is_full = false;
};

#endif