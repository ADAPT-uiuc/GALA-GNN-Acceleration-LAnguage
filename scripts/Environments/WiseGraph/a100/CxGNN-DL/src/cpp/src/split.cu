#include <string>
#include <vector>

#include "file_op.h"
#include "split.h"
using std::string;

std::vector<Index> Split::cluster;

Split::Split(Yaml::Node &config, std::string split, int out_upper_limit,
             bool shuffle)
    : upper_limit(out_upper_limit), shuffle(shuffle) {
  string dataset_path = config["dataset"]["path"].As<string>();
  string split_type = config["dataset"]["split_type"].As<string>();
  bool is_cluster = config["sampler"][split]["name"].As<string>() == "cluster";
  string split_idx_path = dataset_path + "/processed/split/" + split_type +
                          "/" + split + "_idx.dat";
  FileHandler split_idx_handler(split_idx_path);
  split_idx_handler.readAllToVec<Index>(split_idx);
  if (is_cluster) {
    string num_node_path = dataset_path + "/processed/num_nodes.txt";
    ASSERT(fexist(num_node_path));
    FILE *fin(fopen(num_node_path.c_str(), "r"));
    Index num_node = 0;
    fscanf(fin, "%lld", &num_node);
    fclose(fin);
    target_set_mask.resize(num_node);
    for (const auto &idx : split_idx) {
      target_set_mask[idx] = true;
    }
    // cluster ptr
    string path = dataset_path + "/processed/cluster_ptr.dat";
    FileHandler cluster_ptr_handler(path);
    split_idx.clear();
    cluster_ptr_handler.readAllToVec<Index>(cluster_ptr);
    // split idx
    split_idx.resize(cluster_ptr.size() - 1);
    std::iota(split_idx.begin(), split_idx.end(), 0);
    // cluster idx
    if (cluster.size() == 0) {
      path = dataset_path + "/processed/cluster.dat";
      FileHandler cluster_handler(path);
      cluster_handler.readAllToVec<Index>(cluster);
    }
  }
  head = 0;
  if (upper_limit >= split_idx.size()) upper_limit = -1;
  if (upper_limit != -1) {
    ASSERT(upper_limit < split_idx.size());
    split_idx.resize(upper_limit);
    SPDLOG_WARN("using upper limit {} {}", split, upper_limit);
  }
  if (shuffle) {
    random_engine.seed(random_seed);
    std::shuffle(split_idx.begin(), split_idx.end(), random_engine);
  }
  // multi server
  if (getenv("SLURM_NTASKS") != nullptr && atoi(getenv("SLURM_NTASKS")) > 1) {
    ASSERT(getenv("SLURM_PROCID") != nullptr);
    int rank = atoi(getenv("SLURM_PROCID"));
    int num_server = atoi(getenv("SLURM_NTASKS"));
    int begin, end;
    int part = split_idx.size() / num_server;
    begin = part * rank;
    end = part * (rank + 1);
    if (rank == num_server - 1) end = split_idx.size();
    split_idx =
        std::vector<Index>(split_idx.begin() + begin, split_idx.begin() + end);
  }
  num_split_node = split_idx.size();
}

Split::Split(std::vector<Index> &&split_idx, bool shuffle)
    : split_idx(split_idx), head(0), shuffle(shuffle) {
  num_split_node = split_idx.size();
  if (shuffle) {
    random_engine.seed(random_seed);
    std::shuffle(split_idx.begin(), split_idx.end(), random_engine);
  }
}

void Split::setRepeat(int out_repeat) {
  ASSERT(out_repeat >= 1 && repeat == 1);
  repeat = out_repeat;
  int origin_size = split_idx.size();
  if (repeat > 1) {
    split_idx.resize(repeat * origin_size);
    for (int i = 1; i < repeat; ++i) {
      memcpy(split_idx.data() + i * origin_size, split_idx.data(),
             origin_size * sizeof(split_idx[0]));
    }
  }
}

void segmentedShuffle(std::vector<Index> &input, int repeat,
                      uint32_t &random_seed, std::mt19937 &random_engine) {
  int seg_size = input.size() / repeat;
  for (int i = 0; i < repeat; ++i) {
    ++random_seed;
    random_engine.seed(random_seed);
    std::shuffle(input.begin() + i * seg_size,
                 input.begin() + (i + 1) * seg_size, random_engine);
  }
}

void Split::reset() {
  if (shuffle) {
    if (repeat == 1) {
      ++random_seed;
      random_engine.seed(random_seed);
      std::shuffle(split_idx.begin(), split_idx.end(), random_engine);
    } else {
      ASSERTWITH(0, "not tested");
      segmentedShuffle(split_idx, repeat, random_seed, random_engine);
    }
  }
  head = 0;
}

std::vector<Index> Split::sample_cluster(Index num_samples) {
  std::vector<Index> sequence;
  SPDLOG_INFO("head={}", head);
  int ret = 0;
  if (num_samples == -1) {  // fetch all nodes
    num_samples = num_split_node;
    ret = fetchAdd(num_samples);
    if (ret != 0) return sequence;  // empty
  } else {
    ret = fetchAdd(num_samples);
    if (ret == -1) return sequence;
    if (ret + num_samples > this->size()) num_samples = this->size() - ret;
  }
  SPDLOG_INFO("num_samples={} ret={} head={}", num_samples, ret, head);
  for (int i = 0; i < num_samples; i++) {
    Index idx = split_idx[ret + i];
    Index begin = cluster_ptr[idx];
    Index end = cluster_ptr[idx + 1];
    Index seq_begin = sequence.size();
    sequence.resize(seq_begin + end - begin);
    memcpy(sequence.data() + seq_begin, cluster.data() + begin,
           (end - begin) * sizeof(Index));
  }
  return sequence;
}