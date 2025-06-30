#include <string>

#include "fastgraph.h"

shared_ptr<Split> FastGraph::getSplit(const std::string &split_name) {
  if (split_name == "train")
    return train_split;
  else if (split_name == "valid")
    return val_split;
  else if (split_name == "test")
    return test_split;
  else
    ASSERTWITH(false, "unknown split {}", split_name);
  return nullptr;
}

shared_ptr<Dataset> FastGraph::getDataset() { return dataset; }

void FastGraph::initLogLevel(Yaml::Node &config) {
  spdlog::set_pattern("[%H-%M-%S-%e] [%n] [thread %t] [%s:%# %!] %v");
  int log_level = config["util"]["log_level"].As<int>(-1);
  ASSERTWITH(log_level >= 0 && log_level <= 6, "log level must be in [0, 6]");
  if (log_level == 0)
    spdlog::set_level(spdlog::level::debug);
  else if (log_level == 1)
    spdlog::set_level(spdlog::level::info);
  else
    spdlog::set_level(spdlog::level::warn);
}

void FastGraph::initSplit(Yaml::Node &config) {
  bool train_shuffle = config["testing"]["train_shuffle"].As<bool>(true);
  bool eval_shuffle = config["testing"]["eval_shuffle"].As<bool>(false);
  int num_train_samples = config["testing"]["num_train_samples"].As<int>(-1);
  train_split = std::make_shared<Split>(config, "train", num_train_samples,
                                        train_shuffle);
  int num_val_samples = config["testing"]["num_val_samples"].As<int>(-1);
  int num_test_samples = config["testing"]["num_test_samples"].As<int>(-1);
  val_split =
      std::make_shared<Split>(config, "valid", num_val_samples, eval_shuffle);
  test_split =
      std::make_shared<Split>(config, "test", num_test_samples, eval_shuffle);
}

void FastGraph::initDataset(Yaml::Node &config) {
  dataset = Dataset::create(config);
}

void FastGraph::initLoaders(Yaml::Node &config) {
  train_loader = AbstractLoader::create(config, dataset, train_split, "train");
  val_loader = AbstractLoader::create(config, dataset, val_split, "eval");
  test_loader = AbstractLoader::create(config, dataset, test_split, "eval");
}

void FastGraph::start(const std::string &split_name) {
  if (split_name == "train")
    loader = train_loader;
  else if (split_name == "valid")
    loader = val_loader;
  else if (split_name == "test")
    loader = test_loader;
  else
    ASSERTWITH(false, "unknown split {}", split_name);
  loader->start();
}

shared_ptr<Batch> FastGraph::get_batch() { return loader->get_batch(); }

FastGraph::FastGraph(const std::string &yaml_filename) {
  Yaml::Node config;
  if (fexist(yaml_filename))
    Yaml::Parse(config, yaml_filename.c_str());
  else
    ASSERTWITH(false, "yaml file {} not found", yaml_filename);
  initLogLevel(config);
  initDataset(config);
  initSplit(config);
  initLoaders(config);
}