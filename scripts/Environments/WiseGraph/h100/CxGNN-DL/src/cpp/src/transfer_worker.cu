#include "parallel.h"
#include "transfer_worker.h"

void TransferWorker::initDevice(Yaml::Node &config) {
  std::string str_device = config["device"].As<std::string>();
  if (str_device == "cpu") {
    device_ = torch::Device(torch::kCPU);
  } else if (str_device.find("cuda") != std::string::npos) {
    device_id_ = stoi(str_device.substr(5));
    device_ = torch::Device(torch::kCUDA, device_id_);
    SPDLOG_INFO("worker device id {}", device_id_);
  } else {
    ASSERTWITH(false, "device str {}", str_device);
  }
  checkCudaErrors(cudaSetDevice(device_id_));
  torch_stream = at::cuda::getStreamFromPool(false, device_id_);
}

TransferWorker::TransferWorker(Yaml::Node &config,
                               Parallelizer<Batch> *parallelizer)
    : parallelizer(parallelizer) {
  // SPDLOG_WARN("transfer worker init");
  initDevice(config);
}

shared_ptr<Batch> TransferWorker::get() { return get_batch(); }

shared_ptr<Batch> TransferWorker::get_batch() {
  auto batch_in = parallelizer->get();
  if (batch_in != nullptr) {
    // SPDLOG_WARN("Transferring batch");
    batch_in->to(device_, device_id_, torch_stream);
    // SPDLOG_WARN("Transferring batch done");
    // SPDLOG_WARN("Converting batch to fp32");
    if (batch_in->x.sizes()[0] != 0)
      batch_in->x = batch_in->x.to(torch::kFloat32);
    // SPDLOG_WARN("Converting batch to fp32 done");
  }
  return batch_in;
}

void MultiTransferWorker::initDevice(Yaml::Node &config) {
  int num_device = config["num_device"].As<int>(1);
  ASSERT(num_device > 1);
  for (int i = 0; i < num_device; ++i) {
    device_ids_.push_back(i);
    devices_.push_back(torch::Device(torch::kCUDA, i));
    checkCudaErrors(cudaSetDevice(i));
    torch_streams.push_back(at::cuda::getStreamFromPool(false, i));
  }
}

MultiTransferWorker::MultiTransferWorker(Yaml::Node &config,
                                         Parallelizer<Batch> *parallelizer)
    : parallelizer(parallelizer) {
  initDevice(config);
}

shared_ptr<Batch> MultiTransferWorker::get() { return get_batch(); }

shared_ptr<Batch> MultiTransferWorker::get_batch() {
  auto batch_in = parallelizer->get();
  if (batch_in != nullptr) {
    batch_in->toDevices(devices_, device_ids_, torch_streams);
    // if (batch_in->x.sizes()[0] != 0) batch_in->x =
    // batch_in->x.to(torch::kFloat32);
  }
  return batch_in;
}