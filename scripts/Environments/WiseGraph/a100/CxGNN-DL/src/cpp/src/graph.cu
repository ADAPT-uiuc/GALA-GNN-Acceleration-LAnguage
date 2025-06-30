#include "common.h"
#include "graph.h"

bool isUnDirected(const std::vector<Index> &edge_index) {
  Index num_edge = edge_index.size() / 2;
  flat_hash_map<std::pair<Index, Index>, int> edge_map;
  Index l, r;
  for (Index i = 0; i < num_edge; ++i) {
    if (edge_index[i] > edge_index[i + num_edge]) {
      l = edge_index[i];
      r = edge_index[i + num_edge];
    } else {
      l = edge_index[i + num_edge];
      r = edge_index[i];
    }
    auto pair = std::make_pair(l, r);
    if (edge_map.find(pair) != edge_map.end()) {
      edge_map[pair]++;
    } else {
      edge_map[pair] = 1;
    }
  }
  bool undirected = true;
  int mmax = 0;
  int mmin = 999;
  for (auto it = edge_map.begin(); it != edge_map.end(); ++it) {
    if (it->second < 2) undirected = false;
    if (it->second > mmax) mmax = it->second;
    if (it->second < mmin) mmin = it->second;
  }
  SPDLOG_WARN("undirected {} mmax {} mmin {}", undirected, mmax, mmin);
  return undirected;
}

Graph::Graph(Yaml::Node &config) : config(config) {
  // basic information
  string dataset_path = config["dataset"]["path"].As<string>();
  // TODO: we should add a preprocessing module in our system
  bool symmetric = config["dataset"]["symmetric"].As<bool>(true),
       transform_to_undirected = false;
  // For fullgraph, current implementation does not need the edge index file (if
  // CSR file is available)
  bool load_edge_index = config["dataset"]["no_edge_index"].As<bool>(false);
  string sampler_name = config["sampler"]["train"]["name"].As<string>();
  bool etype_schedule =
      config["compute_schedule"]["etype_schedule"].As<bool>(false);
  bool load_edge_type = false;
  if (sampler_name == "typed_neighbor" || etype_schedule) {
    SPDLOG_WARN("loading edge types");
    load_edge_type = true;
  }
  else {
    SPDLOG_WARN("not loading edge types, sampler name {}", sampler_name);
  }
  string subfix = "directed";
  if (symmetric) subfix = "undirected";
  string csr_ptr_path = dataset_path + "/processed/csr_ptr_" + subfix + ".dat";
  string csr_idx_path = dataset_path + "/processed/csr_idx_" + subfix + ".dat";
  if (!fexist(csr_ptr_path) || !fexist(csr_idx_path)) load_edge_index = true;
  // load edge index
  if (load_edge_index) {
    string edge_idx_path = dataset_path + "/processed/edge_index.dat";
    if (symmetric) {
      string ud_edge_idx_path = dataset_path + "/processed/ud_edge_index.dat";
      if (fexist(ud_edge_idx_path)) {
        SPDLOG_INFO("Load undirected edge index from {}", ud_edge_idx_path);
        edge_idx_path = ud_edge_idx_path;
      } else
        transform_to_undirected = true;
    }
    SPDLOG_WARN("Read edge index file");
    FileHandler edge_idx_handler(edge_idx_path);
    edge_idx_handler.readAllToVec<Index>(edge_idx);
    // std::vector<Index> tmp = {0, 1, 1, 0};
    // isUnDirected(tmp);
    if (transform_to_undirected) {
      SPDLOG_INFO("Transform to undirected graph");
      toUndirected();
    }
    SPDLOG_WARN("Done");
    num_edge_ = edge_idx.size() / 2;
    coo_avail_ = true;
  }

  // number of nodes
  string num_node_path = dataset_path + "/processed/num_nodes.txt";
  ASSERT(fexist(num_node_path));
  FILE *fin(fopen(num_node_path.c_str(), "r"));
  fscanf(fin, "%lld", &num_node_);
  fclose(fin);

  // load label
  string label_path = dataset_path + "/processed/node_labels.dat";
  if (!fexist(label_path)) {
    SPDLOG_WARN("Synthetic label {}", num_node_);
    label.resize(num_node_);
    uint32_t random_seed = 233;
    for (Index i = 0; i < num_node_; ++i)
      label[i] = (Index)randInt(0, 20, random_seed);
  } else {
    FileHandler label_handler(label_path);
    label_handler.readAllToVec<Index>(label);
  }
  if (load_edge_type) {
    string etype_path = dataset_path + "/processed/edge_types.dat";
    ASSERT(fexist(etype_path));
    FileHandler etype_handler(etype_path);
    etype_handler.readAllToVec<EtypeIndex>(edge_type);
    string num_etype_path = dataset_path + "/processed/num_etypes.txt";
    ASSERT(fexist(num_etype_path));
    FILE *fin(fopen(num_etype_path.c_str(), "r"));
    fscanf(fin, "%ld", &num_etype_);
    fclose(fin);
    ASSERTWITH(num_etype_ > 0, "num_etype_ should be positive {}", num_etype_);
  }
  // load CSR
  SPDLOG_WARN("Read/Dump CSR file");
  if (!fexist(csr_ptr_path) || !fexist(csr_idx_path)) {
    toCSR();
    FileHandler csr_ptr_handler(csr_ptr_path, FileType::mmap_write);
    csr_ptr_handler.writeFile(csr_ptr.size() * sizeof(Index), csr_ptr.data());
    csr_ptr_handler.closeFile();
    FileHandler csr_idx_handler(csr_idx_path, FileType::mmap_write);
    csr_idx_handler.writeFile(csr_idx.size() * sizeof(Index), csr_idx.data());
    csr_idx_handler.closeFile();
  } else {
    FileHandler csr_ptr_handler(csr_ptr_path);
    csr_ptr_handler.readAllToVec<Index>(csr_ptr);
    FileHandler csr_idx_handler(csr_idx_path);
    csr_idx_handler.readAllToVec<Index>(csr_idx);
    num_edge_ = csr_idx.size();
  }
  SPDLOG_WARN("{} {} {} {}", csr_ptr.size(), csr_ptr.back(), csr_idx.size(),
              csr_idx.back());
  subgraph_index = SubgraphIndex(num_node_, true);
  csr_avail_ = true;
}

Graph::Graph(std::vector<Index> &&out_edge_idx)
    : edge_idx(std::move(out_edge_idx)) {
  num_edge_ = edge_idx.size() / 2;
  coo_avail_ = true;
}

Graph::Graph(std::vector<Index> &&out_csr_ptr, std::vector<Index> &&out_csr_idx,
             std::vector<Index> &&out_num_node_in_layer)
    : csr_ptr(std::move(out_csr_ptr)),
      csr_idx(std::move(out_csr_idx)),
      num_node_in_layer(std::move(out_num_node_in_layer)) {
  num_node_ = csr_ptr.size() - 1;
  num_edge_ = csr_idx.size();
  csr_avail_ = true;
}

void Graph::toCOO() {
  ASSERT(hasCSR());
  // set edge idx
  edge_idx.resize(csr_idx.size() * 2);  // [2*E]
  Index num_edge = csr_idx.size();
  for (Index i = 0; i < num_node_; ++i) {
    for (Index j = csr_ptr[i]; j < csr_ptr[i + 1]; ++j) {
      edge_idx[j + num_edge] = i;
    }
  }
  memcpy(edge_idx.data(), csr_idx.data(),
         sizeof(Index) * csr_idx.size());  // dst
  coo_avail_ = true;
}

void Graph::toCSR() {
  ASSERT(hasCOO());
  ASSERTWITH(num_node_ > 0, "num node should be larger than 0, but is {}",
             num_node_);
  std::vector<Index> degree(num_node_, 0);
  for (Index i = num_edge_; i < num_edge_ * 2; ++i) {
    ++degree[edge_idx[i]];  // dst degree
  }
  csr_ptr = std::vector<Index>(num_node_ + 1);
  csr_ptr[0] = 0;
  csr_idx = std::vector<Index>(num_edge_);
  for (Index i = 0; i < num_node_; ++i) {
    csr_ptr[i + 1] = csr_ptr[i] + degree[i];
  }
  memset(degree.data(), 0, num_node_ * sizeof(Index));
  for (Index i = 0; i < num_edge_; ++i) {
    Index dst = edge_idx[i + num_edge_];
    csr_idx[csr_ptr[dst] + degree[dst]] = edge_idx[i];
    ++degree[dst];
  }
  csr_avail_ = true;
}

void Graph::toUndirected() {
  Index num = edge_idx.size() / 2;
  edge_idx.resize(edge_idx.size() * 2);
  memcpy(edge_idx.data() + 2 * num, edge_idx.data() + num, num * sizeof(Index));
  memcpy(edge_idx.data() + 3 * num, edge_idx.data(), num * sizeof(Index));
}

shared_ptr<Graph> Graph::induce(const SubgraphIndex &subgraph_index) {
  std::vector<Index> new_ptr(subgraph_index.sub_to_full.size() + 1, 0);
  std::vector<Index> new_idx;  // size is unknown
  ASSERT(this->hasCSR());
  for (Index i = 0; i < subgraph_index.sub_to_full.size(); ++i) {
    Index node = subgraph_index.sub_to_full[i];
    Index begin = this->csr_ptr[node], end = this->csr_ptr[node + 1];
    Index cnt = 0;
    for (Index j = begin; j < end; ++j) {
      auto iter = subgraph_index.full_to_sub.find(this->csr_idx[j]);
      if (iter != subgraph_index.full_to_sub.end()) {
        ++cnt;
        new_idx.push_back(iter->second);
      }
    }
    new_ptr[i + 1] = new_ptr[i] + cnt;
  }
  auto graph = make_shared<Graph>(std::move(new_ptr), std::move(new_idx));
  graph->setSubgraphIdx(subgraph_index);
  return graph;
}

// shared_ptr<Graph> Graph::induceLayered(
//     const SubgraphIndex &subgraph_index, Index num_seed, int num_layer,
//     const std::vector<Index> &num_node_in_layer) {
//   std::vector<Index> new_ptr(subgraph_index.sub_to_full.size() + 1, 0);
//   std::vector<Index> new_idx;  // size is unknown
//   ASSERT(this->hasCSR());
//   std::vector<Index> new_num_node_in_layer(num_layer + 1, 0);
//   std::vector<Index> new_num_edge_in_layer(num_layer, 0);
//   new_num_node_in_layer[0] = num_seed;
//   Index layered_cnt = num_seed;
//   int layer_curr = 0;
//   for (Index i = 0; i < subgraph_index.sub_to_full.size(); ++i) {
//     Index node = subgraph_index.sub_to_full[i];
//     Index begin = this->csr_ptr[node], end = this->csr_ptr[node + 1];
//     Index cnt = 0;
//     for (Index j = begin; j < end; ++j) {
//       auto iter = subgraph_index.full_to_sub.find(this->csr_idx[j]);
//       if (iter != subgraph_index.full_to_sub.end()) {
//         ++cnt;
//         new_idx.push_back(iter->second);
//       }
//     }
//     new_ptr[i + 1] = new_ptr[i] + cnt;
//     if (i == layered_cnt - 1) {
//       new_num_node_in_layer[layer_curr + 1] = ;
//     }
//   }
//   auto graph = make_shared<Graph>(std::move(new_ptr), std::move(new_idx));
//   graph->setSubgraphIdx(subgraph_index);
//   return graph;
// }