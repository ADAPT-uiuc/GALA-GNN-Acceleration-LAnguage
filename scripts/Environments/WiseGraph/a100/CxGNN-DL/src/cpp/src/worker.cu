#include <linux/stat.h>
#include <pthread.h>
#include <pybind11/pybind11.h>
#include <sched.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include "Yaml.hpp"
#include "common.h"
#include "compute_schedule.h"
#include "parallel.h"
#include "worker.h"

// #define PROF
std::pair<Tensor, Tensor> BatchWorker::gen_xy_from_graph(
    shared_ptr<Graph> subgraph, std::vector<Index> &mask_in_full) {
  Index feature_len = dataset_->feature_len_;
  int num_unique = subgraph->subgraph_index.num_nodes;
  int num_label = mask_in_full.size();
  // Performance improved: allocate tensor instead of vector
  Tensor x;
  if (load_x_) {
    x = torch::empty({num_unique, feature_len}, dataset_->feature_option);
    if (subgraph->subgraph_index.is_full)
      dataset_->fetchAllData(x);
    else
      dataset_->fetchData(subgraph->subgraph_index.sub_to_full, feature_len, x);
  }
  Tensor y = torch::empty({num_label}, int64_option);
  for (int i = 0; i < mask_in_full.size(); ++i)
    y.data<int64_t>()[i] = dataset_->graph_->label[mask_in_full[i]];
  std::pair<Tensor, Tensor> ret = std::make_pair(x, y);
  return ret;
}

shared_ptr<Batch> gen_batch_from_graph(Tensor x, Tensor y,
                                       shared_ptr<Graph> subgraph,
                                       std::vector<Index> &mask_in_sub,
                                       GraphType graph_type) {
  shared_ptr<Batch> batch;
  if (graph_type == GraphType::COO) {
    int num_edge = subgraph->edge_idx.size() / 2;
    batch = make_shared<Batch>(
        x, y,
        torch::from_blob(subgraph->edge_idx.data(), {2, num_edge},
                         int64_option),
        torch::from_blob(mask_in_sub.data(), {(Index)mask_in_sub.size()},
                         int64_option));

    batch->setNodeInLayer(subgraph->num_node_in_layer);
    batch->setEdgeInLayer(subgraph->num_edge_in_layer);
    batch->setEtypeInLayer(subgraph->num_etype_in_layer);
    batch->setSubToFull(subgraph->subgraph_index.sub_to_full);
  } else {
    batch = make_shared<Batch>(
        x, y, torch::Tensor(),
        torch::from_blob(mask_in_sub.data(), {(Index)mask_in_sub.size()},
                         int64_option),
        torch::from_blob(subgraph->subgraph_index.sub_to_full.data(),
                         {(Index)subgraph->subgraph_index.sub_to_full.size()},
                         int64_option));
    int num_edge = subgraph->getNumEdge();
    ASSERTWITH(subgraph->hasCSR(), "do not have CSR");
    // SPDLOG_WARN("{} {} {}", num_edge, subgraph->csr_ptr.size(),
    // subgraph->csr_idx.size());
    ASSERTWITH(
        subgraph->csr_ptr.size() > 0 /*&& subgraph->csr_idx.size() > 0 */,
        "csr {} {}", subgraph->csr_ptr.size(), subgraph->csr_idx.size());
    int len_ptr = subgraph->csr_ptr.size();
    batch->setCSR(
        torch::from_blob(subgraph->csr_ptr.data(), {len_ptr}, int64_option),
        torch::from_blob(subgraph->csr_idx.data(), {num_edge}, int64_option));
    if (graph_type != GraphType::COO) {
      batch->setNodeInLayer(subgraph->num_node_in_layer);
      batch->setEdgeInLayer(subgraph->num_edge_in_layer);
      batch->setEtypeInLayer(subgraph->num_etype_in_layer);
      batch->graph_type = graph_type;
    }
  }
  if (subgraph->edge_type.size() > 0) {
    batch->setEdgeType(torch::from_blob(subgraph->edge_type.data(),
                                        {(Index)subgraph->edge_type.size()},
                                        int32_option));
  }
  batch->subgraph = subgraph;
  batch->setMaskInSub(std::move(mask_in_sub));  // make its life longer
  return batch;
}

BatchWorker::BatchWorker(Yaml::Node &config, shared_ptr<Dataset> dataset,
                         const std::string &name)
    : dataset_(dataset) {
  initSampler(config["sampler"][name]);
  initDevice(config);
  initGraphType(config);
  initScheduler(config);
  auto feat_mode = config["loading"]["feat_mode"].As<std::string>();
  load_x_ = feat_mode == "memory" || feat_mode == "disk";
}

shared_ptr<Batch> BatchWorker::get() { return get_batch(); }

shared_ptr<Batch> BatchWorker::get_batch() {
  SamplerReturnType sampler_ret =
      sampler_->sample(dataset_->graph_, output_graph_type_);
  shared_ptr<Graph> subgraph = std::get<0>(sampler_ret);
  if (subgraph == nullptr) return nullptr;
  // ID of the nodes with label (seed nodes) in full graph
  std::vector<Index> &mask_in_full = std::get<1>(sampler_ret);
  // ID of the nodes with label (seed nodes) in subgraph
  std::vector<Index> &mask_in_sub = std::get<2>(sampler_ret);
  if (scheduler_ != nullptr) {
    scheduler_->schedule(subgraph, subgraph->num_node_in_layer.size() - 2);
  }
  std::vector<torch::Tensor> etype_partition;
  if (subgraph->edge_type.size() != 0) {
    etype_partition = rel_schedule(subgraph, subgraph->edge_type);
  }
  auto p = gen_xy_from_graph(subgraph, mask_in_full);
  shared_ptr<Batch> batch = gen_batch_from_graph(
      p.first, p.second, subgraph, mask_in_sub, output_graph_type_);
  if (etype_partition.size() != 0) {
    batch->setEtypePartition(std::move(etype_partition));
  }
  return batch;
}

void BatchWorker::set_split(shared_ptr<Split> split) {
  sampler_->set_split(split);
}

void BatchWorker::initGraphType(Yaml::Node &config) {
  std::string output_type = config["output"]["graph_type"].As<string>();
  if (output_type == "COO")
    output_graph_type_ = GraphType::COO;
  else if (output_type == "CSR")
    output_graph_type_ = GraphType::CSR;
  else if (output_type == "CSR_Layer")
    output_graph_type_ = GraphType::CSR_Layer;
  else if (output_type == "CSR_Schedule")
    output_graph_type_ = GraphType::CSR_Schedule;
  else {
    ASSERTWITH(0, "wrong output type {}", output_type);
  }
}

void BatchWorker::initSampler(Yaml::Node &config) {
  sampler_ = Sampler::create(config);
}

void BatchWorker::initDevice(Yaml::Node &config) {
  std::string str_device = config["device"].As<std::string>();
  if (str_device == "cpu") {
    device_ = torch::Device(torch::kCPU);
  } else if (str_device.find("cuda") != std::string::npos) {
    device_id_ = stoi(str_device.substr(5));
    device_ = torch::Device(torch::kCUDA, device_id_);
    // SPDLOG_INFO("worker device id {}", device_id_);
  } else {
    ASSERTWITH(false, "device str {}", str_device);
  }
  checkCudaErrors(cudaSetDevice(device_id_));
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  torch_stream = at::cuda::getStreamFromPool(false, device_id_);
}

void BatchWorker::initScheduler(Yaml::Node &config) {
  if (output_graph_type_ == GraphType::CSR_Schedule)
    scheduler_ = ComputeScheduler::create(config);
}