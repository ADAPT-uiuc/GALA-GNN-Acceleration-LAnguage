#ifndef MEMORY_ACCESS_H
#define MEMORY_ACCESS_H
#include <torch/torch.h>

#include "common.h"

torch::Tensor uvm_select(torch::Tensor buffer, torch::Tensor index);

torch::Tensor uvm_select_masked(torch::Tensor buffer, torch::Tensor index,
                                torch::Tensor mask);

torch::Tensor uvm_select_half(torch::Tensor buffer, torch::Tensor index);

torch::Tensor uvm_select_masked_half(torch::Tensor buffer, torch::Tensor index,
                                     torch::Tensor mask);

torch::Tensor gen_mmap(std::string path, int feature_len, int data_length,
                       bool set_random);

torch::Tensor mmap_select(torch::Tensor buffer, torch::Tensor index);

torch::Tensor mmap_select_st(torch::Tensor buffer, torch::Tensor index);

void read_to_ptr(int64_t ptr, std::string path, int64_t size);

std::vector<torch::Tensor> graph_analysis(torch::Tensor ptr, torch::Tensor idx,
                                          std::vector<Index> train_nid,
                                          Index num_node, int num_layer);
#endif