#include <linux/stat.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <tuple>

#include "Yaml.hpp"
#include "common.h"
#include "loader.h"
#include "worker.h"

namespace py = pybind11;
using std::get;

shared_ptr<AbstractLoader> AbstractLoader::create(Yaml::Node &config,
                                                  shared_ptr<Dataset> dataset,
                                                  shared_ptr<Split> split,
                                                  std::string name) {
  return std::make_shared<NaiveLoader>(config, dataset, split, name);
}

NaiveLoader::NaiveLoader(Yaml::Node &config, shared_ptr<Dataset> dataset,
                         shared_ptr<Split> split, std::string name)
    : AbstractLoader(config, dataset, split, name) {
  initMultiThread(config);
  initTransferer(config);
  Index batch_size = config["sampler"][name]["batch_size"].As<Index>(1);
  num_iters = (split->num_split_node + batch_size - 1) / batch_size;
}

void NaiveLoader::initMultiThread(Yaml::Node &config) {
  num_threads = config["performance"]["num_thread"].As<int>();
  int max_in_flight = config["performance"]["max_in_flight"].As<int>(-1);
  int bind_method =
      config["performance"]["bind_method"].As<int>(0);  // default no bind
  ASSERT(max_in_flight != -1);
  for (int i = 0; i < num_threads; ++i) {
    worker_vec_.push_back(new BatchWorker(config, dataset, name));
    worker_vec_[i]->set_split(split);
  }
  parallelizer = new Parallelizer<Batch>(
      num_threads, max_in_flight,
      std::vector<Worker<Batch> *>(worker_vec_.begin(), worker_vec_.end()),
      bind_method);
}

void NaiveLoader::initTransferer(Yaml::Node &config) {
  transfer_num_threads =
      config["performance"]["transfer"]["num_thread"].As<int>(-1);
  int transfer_max_in_flight =
      config["performance"]["transfer"]["max_in_flight"].As<int>(-1);
  int bind_method =
      config["performance"]["bind_method"].As<int>(0);  // default no bind
  int num_device = config["num_device"].As<int>(1);
  ASSERT(transfer_num_threads != -1);
  ASSERT(transfer_max_in_flight != -1);
  if (num_device == 1)
    for (int i = 0; i < transfer_num_threads; ++i)
      transfer_worker_vec_.push_back(new TransferWorker(config, parallelizer));
  else if (num_device > 1)
    for (int i = 0; i < transfer_num_threads; ++i)
      transfer_worker_vec_.push_back(
          new MultiTransferWorker(config, parallelizer));
  else {
    ASSERT(0);
  }
  transfer_parallelizer = new Parallelizer<Batch>(
      transfer_num_threads, transfer_max_in_flight,
      std::vector<Worker<Batch> *>(transfer_worker_vec_.begin(),
                                   transfer_worker_vec_.end()),
      bind_method, num_threads);
}

shared_ptr<Batch> NaiveLoader::get_batch() {
  shared_ptr<Batch> batch = transfer_parallelizer->get();
  if (batch) return batch;
  throw py::stop_iteration();
}

void NaiveLoader::start() {
  // 1. Stop the current loader
  transfer_parallelizer->stop();  // transfer depends on sample, stop first
  parallelizer->stop();

  // 2. Reset split
  split->reset();

  // 3. Start working
  parallelizer->start();
  transfer_parallelizer->start();
}

NaiveLoader::~NaiveLoader() {
  if (parallelizer != nullptr) {
    parallelizer->stop();
    delete parallelizer;
    parallelizer = nullptr;
  }
  if (transfer_parallelizer != nullptr) {
    transfer_parallelizer->stop();
    delete transfer_parallelizer;
    transfer_parallelizer = nullptr;
    for (int i = 0; i < transfer_num_threads; ++i) {
      delete transfer_worker_vec_[i];
      transfer_worker_vec_[i] = nullptr;
    }
  }
  for (int i = 0; i < num_threads; ++i) {
    delete worker_vec_[i];
    worker_vec_[i] = nullptr;
  }
}
