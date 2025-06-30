#ifndef TRANSFER_WORKER
#define TRANSFER_WORKER
#include "worker.h"

class TransferWorker : public Worker<Batch> {
 public:
  TransferWorker(Yaml::Node &config, Parallelizer<Batch> *parallelizer);
  void initDevice(Yaml::Node &config);
  shared_ptr<Batch> get() override;
  shared_ptr<Batch> get_batch();

  Parallelizer<Batch> *parallelizer;
  torch::Device device_ = torch::Device(torch::kCUDA, 0);
  int device_id_ = 0;
  at::cuda::CUDAStream torch_stream = at::cuda::getDefaultCUDAStream();
};

class MultiTransferWorker : public Worker<Batch> {
 public:
  MultiTransferWorker(Yaml::Node &config, Parallelizer<Batch> *parallelizer);
  void initDevice(Yaml::Node &config);
  shared_ptr<Batch> get() override;
  shared_ptr<Batch> get_batch();

  Parallelizer<Batch> *parallelizer;
  std::vector<torch::Device> devices_;
  std::vector<int> device_ids_;
  std::vector<at::cuda::CUDAStream> torch_streams;
};

#endif