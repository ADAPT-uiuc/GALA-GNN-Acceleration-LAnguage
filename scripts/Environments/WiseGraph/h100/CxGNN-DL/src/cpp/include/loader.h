#ifndef __LOADER_H__
#define __LOADER_H__

#include <memory>
#include <string>
#include <vector>

#include "Yaml.hpp"
#include "batch.h"
#include "dataset.h"
#include "parallel.h"
#include "queue.h"
#include "sample.h"
#include "split.h"
#include "transfer_worker.h"
#include "worker.h"

enum class LoaderType { Naive, TwoStage };

class AbstractLoader {
 public:
  AbstractLoader(Yaml::Node &config, shared_ptr<Dataset> dataset,
                 shared_ptr<Split> split, std::string name)
      : dataset(dataset), split(split), name(name) {}
  virtual void start() = 0;
  virtual shared_ptr<Batch> get_batch() = 0;
  static shared_ptr<AbstractLoader> create(Yaml::Node &config,
                                           shared_ptr<Dataset> dataset,
                                           shared_ptr<Split> split,
                                           std::string name);
  Index num_iters = 0;

 protected:
  shared_ptr<Dataset> dataset;
  shared_ptr<Split> split;
  std::string name;
};

class NaiveLoader : public AbstractLoader {
 public:
  NaiveLoader(Yaml::Node &config, shared_ptr<Dataset> dataset,
              shared_ptr<Split> split, std::string name);
  void initMultiThread(Yaml::Node &config);
  void initTransferer(Yaml::Node &config);
  shared_ptr<Batch> get_batch() override;
  void start() override;
  ~NaiveLoader();

  // property
  const LoaderType type = LoaderType::Naive;

 private:
  // multi-thread
  int transfer_num_threads;
  std::vector<Worker<Batch> *> transfer_worker_vec_;
  Parallelizer<Batch> *transfer_parallelizer = nullptr;

  int num_threads = 0;
  std::vector<BatchWorker *> worker_vec_;
  Parallelizer<Batch> *parallelizer = nullptr;
};

#endif  // __LOADER_H__