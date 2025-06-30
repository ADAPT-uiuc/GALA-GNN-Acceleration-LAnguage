#ifndef __FASTGRAPH_H__
#define __FASTGRAPH_H__
#include "Yaml.hpp"
#include "common.h"
#include "dataset.h"
#include "loader.h"
#include "split.h"

class FastGraph {
  // This is the top-level class which will be binded to the python module.
 public:
  FastGraph(const std::string &yaml_filename);
  shared_ptr<Batch> get_batch();
  shared_ptr<AbstractLoader> train_loader, val_loader, test_loader;
  void start(const std::string &split_name);
  shared_ptr<Split> getSplit(const std::string &split_name);
  shared_ptr<Dataset> getDataset();
  Index num_iters() { return loader->num_iters; }

 private:
  void initLogLevel(Yaml::Node &config);
  void initSplit(Yaml::Node &config);
  void initDataset(Yaml::Node &config);
  void initLoaders(Yaml::Node &config);
  // resources
  shared_ptr<Dataset> dataset;
  shared_ptr<Split> train_split, val_split, test_split;

  // loader
  shared_ptr<AbstractLoader> loader = nullptr;
};
#endif  // __FASTGRAPH_H__