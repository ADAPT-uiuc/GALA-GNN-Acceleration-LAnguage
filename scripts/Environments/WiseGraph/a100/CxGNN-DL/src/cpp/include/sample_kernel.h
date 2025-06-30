#pragma once
#include <torch/extension.h>
#include <torch/torch.h>

#include "common.h"
#include "subgraph_index.h"

using torch::Tensor;
std::vector<Tensor> neighbor_sample(Tensor ptr, Tensor idx,
                                    const std::vector<int>& fanouts,
                                    Tensor seed_nodes);