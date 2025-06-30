#ifndef DATASET_H
#define DATASET_H

#include <assert.h>

#include <memory>
#include <string>

#include "Yaml.hpp"
#include "common.h"
#include "file_op.h"
#include "graph.h"
#include "spdlog/spdlog.h"
#include "split.h"

using std::make_shared;
using std::shared_ptr;
using std::string;
using torch::Tensor;

class Dataset {
 public:
  Index feature_len_ = 0;
  shared_ptr<Graph> graph_;
  shared_ptr<FileHandler> feature_handler_;  // for node feature

  torch::TensorOptions feature_option;

  static shared_ptr<Dataset> create(Yaml::Node &config);
  virtual void fetchData(const std::vector<Index> &index, int64_t feature_len,
                         Tensor &buf);
  virtual void fetchAllData(Tensor &buf);
  virtual Tensor index2D(const Tensor &tensor, const std::vector<Index> &index);
};

template <class T>
class DatasetTemplate : public Dataset {
 public:
  DatasetTemplate(Yaml::Node &config) {
    graph_ = make_shared<Graph>(config);
    // basic information
    string dataset_path = config["dataset"]["path"].As<string>();
    string split_type = config["dataset"]["split_type"].As<string>();
    // open feature file
    string feature_path = dataset_path + "/processed/node_features.dat";
    string feat_mode = config["loading"]["feat_mode"].As<string>();
    FileType read_mode = FileType::empty;
    if (feat_mode == "empty" || feat_mode == "uvm" ||
        feat_mode == "history_uvm" || feat_mode == "history_mmap" ||
        feat_mode == "random" || feat_mode == "mmap") {
      read_mode = FileType::empty;
      SPDLOG_WARN("feat_mode=empty, no feature access using CPU");
    } else if (feat_mode == "memory") {
      read_mode = FileType::memory;
      SPDLOG_WARN("feat_mode=memory, loading node features from {}",
                  feature_path);
    }
    // else if (feat_mode == "mmap")
    //   read_mode = FileType::mmap;
    else {
      ASSERTWITH(false, "feat_mode is {}", feat_mode);
    }
    feature_handler_ = make_shared<FileHandler>(feature_path, read_mode);
    feature_handler_->openFile();
    feature_len_ =
        feature_handler_->filesize_ / sizeof(T) / graph_->getNumNode();
  }

  void fetchData(const std::vector<Index> &index, int64_t feature_len,
                 Tensor &buf) override {
    feature_handler_->fetchData<T>(index, feature_len, buf);
  }

  void fetchAllData(Tensor &buf) override {
    feature_handler_->fetchAllData<T>(buf);
  }

  Tensor index2D(const Tensor &tensor,
                 const std::vector<Index> &index) override {
    ASSERTWITH(tensor.dim() == 2, "tensor should be 2D");
    int xdim = index.size(), ydim = tensor.size(1);
    // SPDLOG_WARN("Allocing xdim {} ydim {} tensor", xdim, ydim);
    Tensor result = torch::empty({xdim, ydim}, tensor.options());
    // SPDLOG_WARN("Successfully alloced");

    // SPDLOG_WARN("Fetching data");
    for (int i = 0; i < xdim; i++)
      memcpy(result.data<T>() + i * ydim, tensor.data<T>() + index[i] * ydim,
             ydim * sizeof(T));
    // SPDLOG_WARN("Done");
    return result;
  }
};

#endif