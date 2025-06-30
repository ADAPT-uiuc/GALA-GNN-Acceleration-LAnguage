#include "batch.h"
#include "common.h"
// #define PROF_TO

Batch::Batch() {}

Batch::Batch(void *x_data, void *y_data, void *edge_index_data, int num_nodes, int num_features,
             int num_edges, int num_labels) {
  x = torch::from_blob(x_data, {num_nodes, num_features}, float32_option);
  y = torch::from_blob(y_data, {num_labels}, int64_option);
  if (num_edges > 0) edge_index = torch::from_blob(edge_index_data, {2, num_edges}, int64_option);
  mask = torch::arange(0, num_labels, 1, int64_option);
}

Batch::Batch(void *x_data, void *y_data, void *edge_index_data, void *mask_data, int num_nodes,
             int num_features, int num_edges, int num_labels) {
  x = torch::from_blob(x_data, {num_nodes, num_features}, float32_option);
  y = torch::from_blob(y_data, {num_labels}, int64_option);
  edge_index = torch::from_blob(edge_index_data, {2, num_edges}, int64_option);
  mask = torch::from_blob(mask_data, {num_labels}, int64_option);
}

Batch::Batch(const Tensor &x, const Tensor &y, const Tensor &edge_index)
    : x(x), y(y), edge_index(edge_index) {
  mask = torch::arange(0, y.size(0), 1, int64_option);
}

Batch::Batch(const Tensor &x, const Tensor &y, const Tensor &edge_index, const Tensor &mask)
    : x(x), y(y), edge_index(edge_index), mask(mask) {}

Batch::Batch(const Tensor &x, const Tensor &y, const Tensor &edge_index, const Tensor &mask,
             const Tensor &sub_to_full)
    : x(x), y(y), edge_index(edge_index), mask(mask), sub_to_full(sub_to_full) {}

vector<Tensor> Batch::createTensors(torch::Device device, const vector<Tensor> &input_vec) {
  torch::TensorOptions device_option = torch::TensorOptions().device(device);
  vector<Tensor> output_vec;
  for (const auto &item : input_vec) {
    // Index size = 1;
    // for (const auto &s : item.sizes())
    //   size *= s;
    // SPDLOG_WARN("{} Byte", size * item.element_size());
    if (item.sizes()[0] == 0)
      output_vec.push_back(Tensor());
    else
      output_vec.push_back(torch::empty_like(item, device_option));
  }
  return output_vec;
}

void Batch::recordTensors(vector<Tensor> &vec, at::cuda::CUDAStream defaultstream) {
  for (int i = 0; i < vec.size(); ++i)
    if (vec[i].sizes()[0] != 0) vec[i].record_stream(defaultstream);
}

void Batch::pinTensors(vector<Tensor> &vec) {
  for (int i = 0; i < vec.size(); ++i) {
    vec[i] = vec[i].pin_memory();
  }
}

void Batch::moveTensors(vector<Tensor> &input_vec, vector<Tensor> &output_vec,
                        cudaStream_t cudastream) {
  for (int i = 0; i < input_vec.size(); ++i) {
    Index size = 1;
    for (const auto &item : input_vec[i].sizes())
      size *= item;
    if (size == 0) continue;
    // SPDLOG_WARN("{} {} Byte {} <- {}", i, size * input_vec[i].element_size(), 
    // output_vec[i].device().index(), input_vec[i].device().index());
    // if (i == 0) continue; // DEBUG: donot need X tensor
    checkCudaErrors(
        cudaMemcpyAsync((void *)output_vec[i].data_ptr(), (void *)input_vec[i].data_ptr(),
                        size * input_vec[i].element_size(), cudaMemcpyHostToDevice, cudastream));
  }
}

void Batch::to(torch::Device device) {
  x = x.pin_memory().to(device, 1);
  y = y.pin_memory().to(device, 1);
  if (graph_type == GraphType::COO) edge_index = edge_index.pin_memory().to(device, 1);
  mask = mask.pin_memory().to(device, 1);
  if (graph_type == GraphType::CSR || graph_type == GraphType::CSR_Layer ||
      graph_type == GraphType::CSR_Schedule) {
    ptr = ptr.pin_memory().to(device, 1);
    idx = idx.pin_memory().to(device, 1);
  }
}

void Batch::to(torch::Device device, int device_id, at::cuda::CUDAStream s) {
  at::cuda::CUDAStreamGuard guard(s);
  auto cudastream = s.stream();
  auto defaultstream = at::cuda::getDefaultCUDAStream();
#ifdef PROF_TO
  timestamp(t0);
#endif
  vector<Tensor> input_vec = {x, y, mask};
  if (graph_type == GraphType::COO)
    input_vec.push_back(edge_index);
  else if (graph_type == GraphType::CSR || graph_type == GraphType::CSR_Layer ||
           graph_type == GraphType::CSR_Schedule) {
    input_vec.push_back(ptr);
    input_vec.push_back(idx);
    input_vec.push_back(sub_to_full);
  }
  if (etype_partition.size() != 0) {
    typed_num_node_in_layer = etype_partition.back();
    etype_partition.pop_back();
    for (auto item : etype_partition)
      input_vec.push_back(item);
  }

  auto output_vec = createTensors(device, input_vec);
  recordTensors(output_vec, defaultstream);

#ifdef PROF_TO
  timestamp(t1);
#endif
  // pinTensors(input_vec);
#ifdef PROF_TO
  timestamp(t2);
#endif
  moveTensors(input_vec, output_vec, cudastream);
  checkCudaErrors(cudaSetDevice(device_id));
  checkCudaErrors(cudaStreamSynchronize(cudastream));

  // Features & Labels
  x = output_vec[0];
  y = output_vec[1];
  mask = output_vec[2];
  int begin = 0;
  // Graph Structure (COO or CSR)
  if (graph_type == GraphType::COO) {
    edge_index = output_vec[3];
    begin = 4;
  }
  else if (graph_type == GraphType::CSR || graph_type == GraphType::CSR_Layer ||
           graph_type == GraphType::CSR_Schedule) {
    ptr = output_vec[3];
    idx = output_vec[4];
    sub_to_full = output_vec[5];
    begin = 6;
  }
  else
    throw std::runtime_error("Unknown graph type");
  // Edge Partition
  int end = begin + etype_partition.size();
  etype_partition.clear();
  for (; begin < end; ++begin) {
    etype_partition.push_back(output_vec[begin]);
  }
  // Edge Type
  if (end < output_vec.size()) {
    edge_type = output_vec.back();
    ASSERT(output_vec.size() == end + 1);
  }

#ifdef PROF_TO
  timestamp(t3);
  SPDLOG_WARN("In-transfer alloc {} pin {} transfer {}", getDuration(t0, t1), getDuration(t1, t2),
              getDuration(t2, t3));
#endif
}

void Batch::toDevices(const std::vector<torch::Device> &devices, const std::vector<int> &device_ids,
                      const std::vector<at::cuda::CUDAStream> &streams) {
  int num_device = devices.size();

#ifdef PROF_TO
  timestamp(t0);
#endif
  for (int i = 0; i < num_device; ++i) {
#ifdef PROF_TO
    timestamp(tt0);
#endif
    at::DeviceGuard g(devices[i]);
    at::cuda::CUDAStreamGuard guard(streams[i]);
    auto cudastream = streams[i].stream();
    auto defaultstream = at::cuda::getDefaultCUDAStream();
#ifdef PROF_TO
    timestamp(tt1);
#endif
    vector<Tensor> input_vec = {x, y, mask};
    if (graph_type == GraphType::COO)
      input_vec.push_back(edge_index);
    else if (graph_type == GraphType::CSR || graph_type == GraphType::CSR_Layer ||
             graph_type == GraphType::CSR_Schedule) {
      input_vec.push_back(ptr);
      input_vec.push_back(idx);
      input_vec.push_back(sub_to_full);
    }
#ifdef PROF_TO
    timestamp(tt2);
#endif
    auto output_vec = createTensors(devices[i], input_vec);
#ifdef PROF_TO
    timestamp(tt3);
#endif
    recordTensors(output_vec, defaultstream);
#ifdef PROF_TO
    timestamp(tt4);
#endif
    moveTensors(input_vec, output_vec, cudastream);
#ifdef PROF_TO
    timestamp(tt5);
#endif
    xs.push_back(output_vec[0]);
    ys.push_back(output_vec[1]);
    masks.push_back(output_vec[2]);
    if (graph_type == GraphType::COO)
      edge_indexs.push_back(output_vec[3]);
    else if (graph_type == GraphType::CSR || graph_type == GraphType::CSR_Layer ||
             graph_type == GraphType::CSR_Schedule) {
      ptrs.push_back(output_vec[3]);
      idxs.push_back(output_vec[4]);
      sub_to_fulls.push_back(output_vec[5]);
    }
    else
      throw std::runtime_error("Unknown graph type");
#ifdef PROF_TO
    timestamp(tt6);
    SPDLOG_WARN("{}: {} {} {} {} {} {}", i, getDuration(tt0, tt1), getDuration(tt1, tt2),
                getDuration(tt2, tt3), getDuration(tt3, tt4), getDuration(tt4, tt5),
                getDuration(tt5, tt6));
#endif
  }

#ifdef PROF_TO
  timestamp(t1);
#endif

  for (int i = 0; i < num_device; ++i) {
    at::DeviceGuard g(devices[i]);
    at::cuda::CUDAStreamGuard guard(streams[i]);
    auto cudastream = streams[i].stream();
    checkCudaErrors(cudaSetDevice(device_ids[i]));
    checkCudaErrors(cudaStreamSynchronize(cudastream));
  }

#ifdef PROF_TO
  timestamp(t2);
  SPDLOG_WARN("In-transfer alloc {} transfer {}", getDuration(t0, t1), getDuration(t1, t2));
#endif
}

void Batch::setNodeInLayer(const std::vector<Index> &out_node_in_layer) {
  int size = out_node_in_layer.size();
  num_node_in_layer = torch::empty({size}, int64_option);
  memcpy(num_node_in_layer.data<Index>(), out_node_in_layer.data(),
         out_node_in_layer.size() * sizeof(Index));
}

void Batch::setEdgeInLayer(const std::vector<Index> &out_edge_in_layer) {
  int size = out_edge_in_layer.size();
  num_edge_in_layer = torch::empty({size}, int64_option);
  memcpy(num_edge_in_layer.data<Index>(), out_edge_in_layer.data(),
         out_edge_in_layer.size() * sizeof(Index));
}

void Batch::setEtypeInLayer(const std::vector<Index> &out_etype_in_layer) {
  int size = out_etype_in_layer.size();
  if (size == 0) return;
  num_etype_in_layer = torch::empty({size}, int64_option);
  memcpy(num_etype_in_layer.data<Index>(), out_etype_in_layer.data(),
         out_etype_in_layer.size() * sizeof(Index));
}

void Batch::setMaskInSub(std::vector<Index> &&out_mask_in_sub) {
  mask_in_sub = std::move(out_mask_in_sub);
}

void Batch::setSubToFull(const std::vector<Index> &out_sub_to_full) {
  sub_to_full = torch::empty({(int)(out_sub_to_full.size())}, int64_option);
  memcpy(sub_to_full.data<Index>(), out_sub_to_full.data(), out_sub_to_full.size() * sizeof(Index));
}

void Batch::setEtypePartition(std::vector<Tensor> &&out_etype_partition) {
  etype_partition = std::move(out_etype_partition);
}

void Batch::setCSR(const Tensor &out_ptr, const Tensor &out_idx) {
  graph_type = GraphType::CSR;
  ptr = out_ptr;
  idx = out_idx;
}

void Batch::setEdgeType(const Tensor &edge_type) { this->edge_type = edge_type; }