#include <assert.h>
#include <cuda_fp16.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>

#include "memory_access.h"
#include "sys/sysinfo.h"
#include "sys/types.h"

template <class T>
__global__ void IndexSelectMultiKernel(const T *const array,
                                       const int64_t num_feat,
                                       const Index *const index,
                                       const int64_t length, T *const out) {
  int64_t out_row = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = index[out_row];
    while (col < num_feat) {
      out[out_row * num_feat + col] = array[in_row * num_feat + col];
      col += blockDim.x;
    }
    out_row += stride;
  }
}

template <class T>
__global__ void MaskedIndexSelectMultiKernel(
    const T *const array, const int64_t num_feat, const Index *const index,
    const int64_t length, const bool *const mask, T *const out) {
  int64_t out_row = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row < length) {
    if (mask[out_row]) {
      int64_t col = threadIdx.x;
      const int64_t in_row = index[out_row];
      while (col < num_feat) {
        out[out_row * num_feat + col] = array[in_row * num_feat + col];
        col += blockDim.x;
      }
    }
    out_row += stride;
  }
}

torch::Tensor uvm_select(torch::Tensor buffer, torch::Tensor index) {
  ASSERT(index.device().index() >= 0);
  checkCudaErrors(cudaSetDevice(index.device().index()));
  Index num_to_select = index.sizes()[0];
  Index num_feat = buffer.sizes()[1];
  torch::Tensor output = torch::empty({num_to_select, num_feat},
                                      float32_option.device(index.device()));

  dim3 block(512, 1);
  while (block.x >= 2 * num_feat) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid((num_to_select + block.y - 1) / block.y);
  IndexSelectMultiKernel<<<grid, block>>>(buffer.data<float>(), num_feat,
                                          index.data<Index>(), num_to_select,
                                          output.data<float>());
  return output;
}

torch::Tensor uvm_select_masked(torch::Tensor buffer, torch::Tensor index,
                                torch::Tensor mask) {
  ASSERT(index.device().index() >= 0);
  checkCudaErrors(cudaSetDevice(index.device().index()));
  Index num_to_select = index.sizes()[0];
  Index num_feat = buffer.sizes()[1];
  torch::Tensor output = torch::zeros({num_to_select, num_feat},
                                      float32_option.device(index.device()));

  dim3 block(512, 1);
  while (block.x >= 2 * num_feat) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid((num_to_select + block.y - 1) / block.y);
  MaskedIndexSelectMultiKernel<<<grid, block>>>(
      buffer.data<float>(), num_feat, index.data<Index>(), num_to_select,
      mask.data<bool>(), output.data<float>());
  return output;
}

torch::Tensor uvm_select_half(torch::Tensor buffer, torch::Tensor index) {
  ASSERT(index.device().index() >= 0);
  checkCudaErrors(cudaSetDevice(index.device().index()));
  Index num_to_select = index.sizes()[0];
  Index num_feat = buffer.sizes()[1];
  torch::Tensor output = torch::empty({num_to_select, num_feat},
                                      float16_option.device(index.device()));

  dim3 block(512, 1);
  while (block.x >= 2 * num_feat) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid((num_to_select + block.y - 1) / block.y);
  IndexSelectMultiKernel<<<grid, block>>>((__half *)buffer.data_ptr(), num_feat,
                                          index.data<Index>(), num_to_select,
                                          (__half *)output.data_ptr());
  return output;
}

torch::Tensor uvm_select_masked_half(torch::Tensor buffer, torch::Tensor index,
                                     torch::Tensor mask) {
  ASSERT(index.device().index() >= 0);
  checkCudaErrors(cudaSetDevice(index.device().index()));
  Index num_to_select = index.sizes()[0];
  Index num_feat = buffer.sizes()[1];
  torch::Tensor output = torch::zeros({num_to_select, num_feat},
                                      float16_option.device(index.device()));

  dim3 block(512, 1);
  while (block.x >= 2 * num_feat) {
    block.x /= 2;
    block.y *= 2;
  }
  const dim3 grid((num_to_select + block.y - 1) / block.y);
  MaskedIndexSelectMultiKernel<<<grid, block>>>(
      (__half *)buffer.data_ptr(), num_feat, index.data<Index>(), num_to_select,
      mask.data<bool>(), (__half *)output.data_ptr());
  return output;
}

torch::Tensor gen_mmap(std::string path, int feature_len, int data_length,
                       bool set_random) {
  auto fd = open(path.c_str(), O_RDONLY, (mode_t)0600);
  struct stat file_info = {0};
  ASSERTWITH(fstat(fd, &file_info) != -1, "error get file size, path {}", path);
  auto mmap_ptr =
      (char *)mmap(0, file_info.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (set_random) {
    ASSERTWITH(0 == madvise(mmap_ptr, (intmax_t)file_info.st_size, MADV_RANDOM),
               "fail madvise {}", path);
  }
  torch::Tensor output;
  Index filesize = (Index)file_info.st_size;
  if (data_length == 32) {
    Index num_node = filesize / sizeof(float) / feature_len;
    output = torch::from_blob((void *)mmap_ptr, {(int)(num_node), feature_len},
                              float32_option);
  } else if (data_length == 16) {
    Index num_node = filesize / 2 / feature_len;
    output = torch::from_blob((void *)mmap_ptr, {(int)(num_node), feature_len},
                              float16_option);
  } else {
    ASSERTWITH("Error length {}, should be 32 or 16", data_length);
  }
  return output;
}

torch::Tensor mmap_select(torch::Tensor buffer, torch::Tensor index) {
  return buffer.index({index});
}

torch::Tensor mmap_select_st(torch::Tensor buffer, torch::Tensor index) {
  int num = index.sizes()[0];
  int dim = buffer.sizes()[1];
  torch::Tensor output = torch::empty({num, dim}, float32_option);
  auto ptr = output.data<float>();
  auto ptr_buffer = buffer.data<float>();
  auto index_ptr = index.data<Index>();
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < dim; j++) {
      ptr[i * dim + j] = ptr_buffer[index_ptr[i] * dim + j];
    }
  }
  return output;
}

void read_to_ptr(int64_t ptr, std::string path, int64_t size) {
  int fd = open(path.c_str(), O_RDONLY);
  ASSERT(fd >= 0);
  int64_t offset = 0;
  while (size > 0) {
    int64_t read_size = pread(fd, (char *)ptr + offset, size, offset);
    ASSERT(read_size > 0);
    size -= read_size;
    offset += read_size;
  }
  close(fd);
}

std::vector<torch::Tensor> graph_analysis(torch::Tensor ptr, torch::Tensor idx,
                                          std::vector<Index> train_nid,
                                          Index num_node, int num_layer) {
  std::vector<Index> new_train_nid;
  std::vector<bool> visit_map(num_node, false);
  for (auto nid : train_nid) {
    visit_map[nid] = true;
  }
  std::vector<torch::Tensor> output;
  for (int layer_id = 0; layer_id < num_layer; ++layer_id) {
    output.push_back(torch::zeros({num_node}, int64_option));
  }
  auto ptr_ptr = ptr.data<Index>();
  auto idx_ptr = idx.data<Index>();
  for (int layer_id = 0; layer_id < num_layer; ++layer_id) {
    new_train_nid = train_nid;
    auto cnt_ptr = output[layer_id].data<Index>();
    for (auto nid : train_nid) {
      for (Index i = ptr_ptr[nid]; i < ptr_ptr[nid + 1]; ++i) {
        if (!visit_map[idx_ptr[i]]) {
          new_train_nid.push_back(idx_ptr[i]);
          visit_map[idx_ptr[i]] = true;
        }
        cnt_ptr[idx_ptr[i]] += 1;
      }
    }
    train_nid = new_train_nid;
    SPDLOG_INFO("layer {}, num {}", layer_id, train_nid.size());
  }
  return output;
}