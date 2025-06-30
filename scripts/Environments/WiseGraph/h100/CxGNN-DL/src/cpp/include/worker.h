#ifndef __WORKER_H__
#define __WORKER_H__

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <memory>
#include <string>

#include "Yaml.hpp"
#include "batch.h"
#include "common.h"
#include "compute_schedule.h"
#include "dataset.h"
#include "phmap.h"
#include "queue.h"
#include "sample.h"
#include "split.h"
using phmap::flat_hash_map;
using phmap::flat_hash_set;
template <typename T>
class Worker {
 public:
  virtual shared_ptr<T> get() = 0;
};

template <typename T>
class Parallelizer;

class BatchWorker : public Worker<Batch> {
 public:
  BatchWorker(Yaml::Node &config, shared_ptr<Dataset> dataset,
              const std::string &name);

  shared_ptr<Batch> get() override;
  shared_ptr<Batch> get_batch();

  void set_split(shared_ptr<Split> split);
  void initSampler(Yaml::Node &config);
  void initDevice(Yaml::Node &config);
  void initGraphType(Yaml::Node &config);
  void initScheduler(Yaml::Node &config);

  // resources:
  std::shared_ptr<Dataset> dataset_;

  torch::Device device_ = torch::Device(torch::kCUDA, 0);
  int device_id_ = 0;
  cudaStream_t stream;
  at::cuda::CUDAStream torch_stream = at::cuda::getDefaultCUDAStream();

  // current sampler, split and batch size
  std::shared_ptr<Sampler> sampler_ = nullptr;

  // output grpah type
  GraphType output_graph_type_ = GraphType::COO;

  // Scheduler
  std::shared_ptr<ComputeScheduler> scheduler_ = nullptr;

 private:
  std::pair<Tensor, Tensor> gen_xy_from_graph(shared_ptr<Graph> subgraph,
                                              std::vector<Index> &mask_in_full);
  bool load_x_ = true;
};

shared_ptr<Batch> gen_batch_from_graph(Tensor x, Tensor y,
                                       shared_ptr<Graph> subgraph,
                                       std::vector<Index> &mask_in_sub,
                                       GraphType graph_type);

#endif  // __WORKER_H__