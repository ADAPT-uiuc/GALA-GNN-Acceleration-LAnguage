//
// Created by damitha on 4/23/24.
//

#ifndef GNN_ACCELERATION_LANGUAGE_CUDA_H
#define GNN_ACCELERATION_LANGUAGE_CUDA_H

#include "common.h"

class CUDAGenerator : public CodeGenerator
{
public:
    CUDAGenerator(GALAContext* context, std::string& outputPath) : CodeGenerator(context, outputPath)
    {
    }

    void initCMake() override
    {
        std::string cmakeCudaBase = "cmake_minimum_required(VERSION 3.1 FATAL_ERROR)\n"
            "project(gala_cuda LANGUAGES CUDA CXX)\n"
            "set(CMAKE_CXX_COMPILER icpx)\n"
            "find_package(Torch REQUIRED)\n"
            "find_package(OpenMP)\n"
            "if (OPENMP_FOUND)\n"
            "    set(OpenMP_CXX_FLAGS \"-fopenmp\")\n"
            "    set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}\")\n"
            "    set(CMAKE_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}\")\n"
            "else ()\n"
            "    message(FATAL_ERROR \"Need OpenMP\")\n"
            "endif ()\n"
            "include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})\n"
            "link_libraries(\"${TORCH_LIBRARIES}\" cudart cusparse)\n"
            "set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -qopt-report=0  -march=native -xCORE-AVX512 -O3 -DICC -restrict\")\n"
            "set(CMAKE_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed -lpthread -lm -ldl\")\n"
            "set(CMAKE_CXX_STANDARD_LIBRARIES \"${CMAKE_CXX_STANDARD_LIBRARIES} -lnuma\")\n"
            "if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)\n"
            "# set(CMAKE_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS} -lmkl_gnu_thread -lgomp\")\n"
            "    set(CMAKE_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS} -lgomp\")\n"
            "elseif (CMAKE_CXX_COMPILER_ID STREQUAL IntelLLVM)\n"
            "    # set(CMAKE_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS} -lmkl_intel_thread -liomp5\")\n"
            "    set(CMAKE_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS} -liomp5\")\n"
            "endif ()\n"
            "include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})\n"
            "link_libraries(\"${TORCH_LIBRARIES}\" cudart cusparse)\n"
            "add_compile_options(-Xcompiler -fopenmp -march=native -O3)\n"
            "add_compile_definitions(GALA_TORCH)\n"
            "add_compile_definitions(GN_1)\n"
            "add_compile_definitions(PT_0)\n"
            "add_compile_definitions(ST_0)\n"
            "add_compile_definitions(A_ALLOC)";
        std::string cmakeExecutable = "add_executable(gala_model gala.cu)\n"
            "target_compile_features(gala_model PRIVATE cxx_std_14)";
        cmakeCode.addCode(cmakeCudaBase);
        cmakeCode.addCode(cmakeExecutable);
    }

    void initKernels(std::vector<CIRNode*>& program) override
    {
        std::string importBase = "#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.\n"
            "#include <cusparse.h>\n"
            "#include <torch/script.h>\n"
            "#include <cmath>\n"
            "#include <iostream>\n"
            "#include <parallel/algorithm>\n"
            "#include <vector>\n"
            "#include <bits/stdc++.h>\n"
            "#include <omp.h>\n"
            "#include <stdlib.h>\n"
            "#include <torch/torch.h>\n"
            "#include \"../src/formats/csrc_matrix.h\"\n"
            "#include \"../src/formats/dense_matrix.h\"\n"
            "#include \"../src/ops/aggregators.h\"\n"
            "#include \"../src/ops/sparse_matrix_ops.h\"\n"
            "#include \"../src/ops/tiling.h\"\n"
            "#include \"../src/utils/mtx_io.h\"\n"
            "#include \"../tests/common.h\"\n";
        importCode.addCode(importBase);


        std::string cudaInitFunctions = "\n"
"#define CUDA_CHECK(func)\\\n\
  do {\\\n\
    cudaError_t status = (func);\\\n\
    if (status != cudaSuccess) {\\\n\
      printf(\"CUDA API failed at line %d with error: %s (%d)\\n\", __LINE__,\\\n\
             cudaGetErrorString(status), status);\\\n\
      exit(EXIT_FAILURE);\\\n\
    }\\\n\
  } while (0)\\\n\
\n\
#define CUSPARSE_CHECK(func)\\\n\
  do {\\\n\
    cusparseStatus_t status = (func);\\\n\
    if (status != CUSPARSE_STATUS_SUCCESS) {\\\n\
      printf(\"CUSPARSE failed at line %d with error: %s (%d)\\n\", __LINE__,\\\n\
             cusparseGetErrorString(status), status);\\\n\
      exit(EXIT_FAILURE);\\\n\
    }\\\n\
  } while (0)";
        kernelCode.addCode(cudaInitFunctions);

        for (int i = 0; i < program.size(); i++)
        {
            CIRNode* outNode = program[i];
            auto oNode = dynamic_cast<ComputeNode*>(outNode);
            if (oNode)
            {

            } else {
                auto loopNode = dynamic_cast<TrainingLoopNode*>(outNode);
                for (int ix = 0; ix < loopNode->getLoopNodeNum(); ix++)
                {
                    CIRNode* inNode = loopNode->getNode(ix);
                    auto cNode = dynamic_cast<ComputeNode*>(inNode);
                    if (cNode->getOpType() == AGGREGATE_NODE)
                    {
                        // TODO: This needs to be generated by the program
                        std::string tempKernelCode = ""
"extern \"C\" __global__ void __launch_bounds__(256)\n\
    default_function_kernel64(float *__restrict__ C,\n\
                              int *__restrict__ J_indptr_data,\n\
                              float *__restrict__ A, float *__restrict__ B,\n\
                              int *__restrict__ J_indices_data, int nrows,\n\
                              int dcols) {\n\
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {\n\
    for (int j = 0;\n\
         j <\n\
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -\n\
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
         ++j) {\n\
      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +\n\
          (((int)blockIdx.y) * 64)) +\n\
         ((int)threadIdx.x))] =\n\
          (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +\n\
               (((int)blockIdx.y) * 64)) +\n\
              ((int)threadIdx.x))] +\n\
           (A[(j +\n\
               J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] *\n\
            B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +\n\
                                                    ((int)threadIdx.y))])] *\n\
                 dcols) +\n\
                (((int)blockIdx.y) * 64)) +\n\
               ((int)threadIdx.x))]));\n\
      C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +\n\
           (((int)blockIdx.y) * 64)) +\n\
          ((int)threadIdx.x)) +\n\
         32)] =\n\
          (C[(((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +\n\
                (((int)blockIdx.y) * 64)) +\n\
               ((int)threadIdx.x)) +\n\
              32)] +\n\
           (A[(j +\n\
               J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] *\n\
            B[((((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +\n\
                                                     ((int)threadIdx.y))])] *\n\
                  dcols) +\n\
                 (((int)blockIdx.y) * 64)) +\n\
                ((int)threadIdx.x)) +\n\
               32)]));\n\
    }\n\
  }\n\
}\n\
extern \"C\" __global__ void __launch_bounds__(256)\n\
    default_function_kernel32(float *__restrict__ C,\n\
                              int *__restrict__ J_indptr_data,\n\
                              float *__restrict__ A, float *__restrict__ B,\n\
                              int *__restrict__ J_indices_data, int nrows,\n\
                              int dcols) {\n\
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {\n\
    for (int j = 0;\n\
         j <\n\
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -\n\
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
         ++j) {\n\
      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +\n\
          (((int)blockIdx.y) * 64)) +\n\
         ((int)threadIdx.x))] =\n\
          (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +\n\
               (((int)blockIdx.y) * 64)) +\n\
              ((int)threadIdx.x))] +\n\
           (A[(j +\n\
               J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] *\n\
            B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +\n\
                                                    ((int)threadIdx.y))])] *\n\
                 dcols) +\n\
                (((int)blockIdx.y) * 64)) +\n\
               ((int)threadIdx.x))]));\n\
    }\n\
  }\n\
}\n\
extern \"C\" __global__ void __launch_bounds__(256)\n\
    default_function_kernel32_offset(float *__restrict__ C,\n\
                                     int *__restrict__ J_indptr_data,\n\
                                     float *__restrict__ A,\n\
                                     float *__restrict__ B,\n\
                                     int *__restrict__ J_indices_data,\n\
                                     int nrows, int dcols, int offset) {\n\
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {\n\
    for (int j = 0;\n\
         j <\n\
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -\n\
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
         ++j) {\n\
      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +\n\
          (((int)blockIdx.y) * 64)) +\n\
         ((int)threadIdx.x)) +\n\
        offset] =\n\
          (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +\n\
               (((int)blockIdx.y) * 64)) +\n\
              ((int)threadIdx.x)) +\n\
             offset] +\n\
           (A[(j +\n\
               J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] *\n\
            B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +\n\
                                                    ((int)threadIdx.y))])] *\n\
                 dcols) +\n\
                (((int)blockIdx.y) * 64)) +\n\
               ((int)threadIdx.x)) +\n\
              offset]));\n\
    }\n\
  }\n\
}\n\
extern \"C\" __global__ void __launch_bounds__(256)\n\
    default_function_kernel_rem(float *__restrict__ C,\n\
                                int *__restrict__ J_indptr_data,\n\
                                float *__restrict__ A, float *__restrict__ B,\n\
                                int *__restrict__ J_indices_data, int nrows,\n\
                                int dcols, int offset) {\n\
  if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {\n\
    for (int j = 0;\n\
         j <\n\
         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -\n\
          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
         ++j) {\n\
      C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +\n\
          (((int)blockIdx.y) * 64)) +\n\
         ((int)threadIdx.x)) +\n\
        offset] =\n\
          (C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +\n\
               (((int)blockIdx.y) * 64)) +\n\
              ((int)threadIdx.x)) +\n\
             offset] +\n\
           (A[(j +\n\
               J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] *\n\
            B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +\n\
                                                    ((int)threadIdx.y))])] *\n\
                 dcols) +\n\
                (((int)blockIdx.y) * 64)) +\n\
               ((int)threadIdx.x)) +\n\
              offset]));\n\
    }\n\
  }\n\
}";
                        kernelCode.addCode(tempKernelCode);

                        // This is the kernel call
                        std::string tempAggrKernelCall = ""
"torch::Tensor gather_forward_gcn(torch::Tensor input_dense,\n\
                               torch::Tensor offset_graph,\n\
                               torch::Tensor columns_graph,\n\
                               torch::Tensor value_graph) {\n\
  auto nrows = offset_graph.numel() - 1;\n\
  auto nvals = columns_graph.numel();\n\
  auto full_iden = input_dense.numel();\n\
  auto dcols = full_iden / nrows;\n\
  // // Dense\n\
  // Input\n\
  float *iden_ptr = input_dense.data_ptr<float>();\n\
  // Output\n\
  auto options = torch::TensorOptions()\n\
                     .dtype(torch::kFloat)\n\
                     .requires_grad(true)\n\
                     .device(torch::kCUDA, 0);\n\
  auto output_dense = torch::zeros({nrows, dcols}, options);\n\
  float *oden_array = output_dense.data_ptr<float>();\n\
  // Sparse\n\
  int *offset_ptr = offset_graph.data_ptr<int>();\n\
  int *col_ptr = columns_graph.data_ptr<int>();\n\
  float *val_ptr = value_graph.data_ptr<float>();\n\
  cudaStream_t stream1, stream2, stream3;\n\
  if ((int)dcols / 64) {\n\
    cudaStreamCreate(&stream1);\n\
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / 64);\n\
    dim3 blockDim(32, 8);\n\
    default_function_kernel64<<<gridDim, blockDim, 0, stream1>>>(\n\
        oden_array, offset_ptr, val_ptr, iden_ptr, col_ptr, nrows, dcols);\n\
    if ((dcols % 64) > 32) {\n\
      cudaStreamCreate(&stream2);\n\
      dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);\n\
      dim3 blockDim_rem(32, 8);\n\
      default_function_kernel32_offset<<<gridDim_rem, blockDim_rem, 0,\n\
                                         stream2>>>(\n\
          oden_array, offset_ptr, val_ptr, iden_ptr, col_ptr, nrows, dcols,\n\
          ((int)dcols / 64) * 64);\n\
      if ((dcols % 32) > 0) {\n\
        cudaStreamCreate(&stream3);\n\
        dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);\n\
        dim3 blockDim_rem(dcols % 32, 8);\n\
        default_function_kernel_rem<<<gridDim_rem, blockDim_rem, 0, stream3>>>(\n\
            oden_array, offset_ptr, val_ptr, iden_ptr, col_ptr, nrows, dcols,\n\
            (((int)dcols / 64) * 64) + 32);\n\
      }\n\
    } else if ((dcols % 64) > 0) {\n\
      cudaStreamCreate(&stream2);\n\
      dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);\n\
      dim3 blockDim_rem(dcols % 64, 8);\n\
      default_function_kernel_rem<<<gridDim_rem, blockDim_rem, 0, stream2>>>(\n\
          oden_array, offset_ptr, val_ptr, iden_ptr, col_ptr, nrows, dcols,\n\
          ((int)dcols / 64) * 64);\n\
    }\n\
  } else {\n\
    if ((int)dcols / 32) {\n\
      cudaStreamCreate(&stream1);\n\
      dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, (int)dcols / 32);\n\
      dim3 blockDim_rem(32, 8);\n\
      default_function_kernel32<<<gridDim_rem, blockDim_rem, 0, stream1>>>(\n\
          oden_array, offset_ptr, val_ptr, iden_ptr, col_ptr, nrows, dcols);\n\
      if ((dcols % 32) > 0) {\n\
        cudaStreamCreate(&stream2);\n\
        dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);\n\
        dim3 blockDim_rem(dcols % 32, 8);\n\
        default_function_kernel_rem<<<gridDim_rem, blockDim_rem, 0, stream2>>>(\n\
            oden_array, offset_ptr, val_ptr, iden_ptr, col_ptr, nrows, dcols,\n\
            (((int)dcols / 32) * 32));\n\
      }\n\
    } else {\n\
      cudaStreamCreate(&stream1);\n\
      dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1, 1);\n\
      dim3 blockDim_rem(dcols % 32, 8);\n\
      default_function_kernel_rem<<<gridDim_rem, blockDim_rem, 0, stream1>>>(\n\
          oden_array, offset_ptr, val_ptr, iden_ptr, col_ptr, nrows, dcols, 0);\n\
    }\n\
  }\n\
  return output_dense;\n\
}";
                        // Adding the kernel call and setting the name
                        kernelCallCode.addCode(tempAggrKernelCall);
                        cNode->setKernelName("gather_forward_gcn");
                    }
                }
            }
        }
    }

    void dataPrep(std::vector<CIRNode*>& program) override
    {
        // TODO make the transfer based on the data and the transformations applied
        std::string tempTransferCode = "torch::Device device(torch::kCUDA);\n\
  int *dA_csrOffsets, *dA_columns, *dL;\n\
  float *dA_values, *dB;\n\
  bool *d_train_mask, *d_valid_mask, *d_test_mask;\n\
\n\
  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets, (nrows + 1) * sizeof(int)));\n\
  CUDA_CHECK(cudaMalloc((void **)&dA_columns, nvals * sizeof(int)));\n\
  CUDA_CHECK(cudaMalloc((void **)&dA_values, nvals * sizeof(float)));\n\
  CUDA_CHECK(cudaMalloc((void **)&dB, (nrows * emb_size) * sizeof(float)));\n\
  CUDA_CHECK(cudaMalloc((void **)&dL, nrows * sizeof(long)));\n\
  CUDA_CHECK(cudaMalloc((void **)&d_train_mask, nrows * sizeof(bool)));\n\
  CUDA_CHECK(cudaMalloc((void **)&d_valid_mask, nrows * sizeof(bool)));\n\
  CUDA_CHECK(cudaMalloc((void **)&d_test_mask, nrows * sizeof(bool)));\n\
\n\
  CUDA_CHECK(cudaMemcpy(dA_csrOffsets, adj.offset_ptr(),\n\
                        (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dA_columns, adj.ids_ptr(), nvals * sizeof(int),\n\
                        cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dA_values, adj.vals_ptr(), nvals * sizeof(float),\n\
                        cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dB, input_emb.vals_ptr(),\n\
                        (nrows * emb_size) * sizeof(float),\n\
                        cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dL, labels.vals_ptr(), nrows * sizeof(long),\n\
                        cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(d_train_mask, train_mask.vals_ptr(),\n\
                        nrows * sizeof(bool), cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(d_valid_mask, valid_mask.vals_ptr(),\n\
                        nrows * sizeof(bool), cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(d_test_mask, test_mask.vals_ptr(), nrows * sizeof(bool),\n\
                        cudaMemcpyHostToDevice));\n\
  auto options_cu_int = torch::TensorOptions()\n\
                            .dtype(torch::kInt)\n\
                            .requires_grad(false)\n\
                            .device(torch::kCUDA, 0);\n\
  torch::Tensor t_offsets =\n\
      torch::from_blob(dA_csrOffsets, {nrows + 1}, options_cu_int);\n\
  torch::Tensor t_cols = torch::from_blob(dA_columns, {nvals}, options_cu_int);\n\
\n\
  auto options_cu_long =\n\
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA, 0);\n\
  torch::Tensor t_labs = torch::from_blob(dL, {nrows}, options_cu_long);\n\
\n\
  auto options_cu_float_grad = torch::TensorOptions()\n\
                                   .dtype(torch::kFloat)\n\
                                   .requires_grad(true)\n\
                                   .device(torch::kCUDA, 0);\n\
  auto options_cu_float_ngrad = torch::TensorOptions()\n\
                                    .dtype(torch::kFloat)\n\
                                    .requires_grad(false)\n\
                                    .device(torch::kCUDA, 0);\n\
  torch::Tensor t_vals =\n\
      torch::from_blob(dA_values, {nvals}, options_cu_float_ngrad);\n\
  torch::Tensor t_iden =\n\
      torch::from_blob(dB, {nrows, emb_size}, options_cu_float_grad);\n\
\n\
  auto options_cu_bool = torch::TensorOptions()\n\
                             .dtype(torch::kBool)\n\
                             .requires_grad(false)\n\
                             .device(torch::kCUDA, 0);\n\
  torch::Tensor t_train_mask =\n\
      torch::from_blob(d_train_mask, {nrows}, options_cu_bool);\n\
  torch::Tensor t_valid_mask =\n\
      torch::from_blob(d_valid_mask, {nrows}, options_cu_bool);\n\
  torch::Tensor t_test_mask =\n\
      torch::from_blob(d_test_mask, {nrows}, options_cu_bool);";
        preCode.addCode(tempTransferCode);

        std::string tempCleanCuda = "CUDA_CHECK(cudaFree(dA_csrOffsets));\n\
  CUDA_CHECK(cudaFree(dA_values));\n\
  CUDA_CHECK(cudaFree(dA_columns));\n\
  CUDA_CHECK(cudaFree(dB));";
        postCode.addCode(tempCleanCuda);

    }
};

#endif //GNN_ACCELERATION_LANGUAGE_CUDA_H
