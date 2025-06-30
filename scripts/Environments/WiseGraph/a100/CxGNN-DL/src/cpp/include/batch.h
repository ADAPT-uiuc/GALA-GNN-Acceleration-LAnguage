#ifndef __BATCH_H__
#define __BATCH_H__

#include "common.h"
#include "cuda_runtime_api.h"
#include "sample.h"
#include "spdlog/spdlog.h"
// #include "spdlog/spdlog.h"
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/torch.h>

using std::vector;
using torch::Tensor;

struct Batch {
  // x is a float32 tensor of shape (num_total_nodes, num_features)
  Tensor x;
  // y is a int64 tensor of shape (num_labels,)
  Tensor y;
  // edge_index is a int64 tensor of shape (2, num_edges)
  Tensor edge_index;
  // mask is a int64 tensor of shape (num_labels,), indicating the indices of
  // the training / testing samples in the batch.
  Tensor mask;
  Tensor sub_to_full;

  Batch();

  // for CSR
  Tensor ptr;
  Tensor idx;
  // for layered CSR
  Tensor num_node_in_layer, num_edge_in_layer, num_etype_in_layer;
  Tensor typed_num_node_in_layer;
  // for hetero graph
  Tensor edge_type;

  std::vector<Tensor> xs, ys, edge_indexs, masks, sub_to_fulls, ptrs, idxs;

  GraphType graph_type = GraphType::COO;

  std::shared_ptr<Graph> subgraph = nullptr;
  vector<Index> mask_in_sub;

  Batch(void *x_data, void *y_data, void *edge_index_data, int num_nodes,
        int num_features, int num_edges, int num_labels);

  Batch(void *x_data, void *y_data, void *edge_index_data, void *mask_data,
        int num_nodes, int num_features, int num_edges, int num_labels);

  Batch(const Tensor &x, const Tensor &y, const Tensor &edge_index);

  Batch(const Tensor &x, const Tensor &y, const Tensor &edge_index,
        const Tensor &mask);

  Batch(const Tensor &x, const Tensor &y, const Tensor &edge_index,
        const Tensor &mask, const Tensor &sub_to_full);

  void to(torch::Device device);
  void to(torch::Device device, int device_id, at::cuda::CUDAStream s);

  void toDevices(const std::vector<torch::Device> &devices,
                 const std::vector<int> &device_ids,
                 const std::vector<at::cuda::CUDAStream> &streams);

  void setNodeInLayer(const vector<Index> &out_node_in_layer);

  void setEdgeInLayer(const vector<Index> &out_edge_in_layer);

  void setEtypeInLayer(const vector<Index> &out_etype_in_layer);

  void setMaskInSub(std::vector<Index> &&out_mask_in_sub);

  void setSubToFull(const std::vector<Index> &out_sub_to_full);

  std::vector<torch::Tensor> etype_partition;
  void setEtypePartition(std::vector<torch::Tensor> &&out_etype_partition);

  void setCSR(const Tensor &ptr, const Tensor &idx);

  void setEdgeType(const Tensor &edge_type);

  void display() {
    SPDLOG_INFO("x {} {} y {} mask {} edgeidex {} {} ptr {} idx {}",
                x.sizes()[0], x.sizes()[1], y.sizes()[0], mask.sizes()[0],
                edge_index.sizes()[0], edge_index.sizes()[1], ptr.sizes()[0],
                idx.sizes()[0]);
    // std::cout << "display batch " << torch::_shape_as_tensor(x) << ' ' <<
    // x.device() << ' '
    //               << mask.device() << std::endl;
  };

  // ~Batch();

  vector<Tensor> createTensors(torch::Device device,
                               const vector<Tensor> &input_vec);
  void recordTensors(vector<Tensor> &vec, at::cuda::CUDAStream defaultstream);
  void pinTensors(vector<Tensor> &vec);
  void moveTensors(vector<Tensor> &input_vec, vector<Tensor> &output_vec,
                   cudaStream_t cudastream);
};

#endif  // __BATCH_H__