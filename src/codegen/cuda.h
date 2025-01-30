//
// Created by damitha on 4/23/24.
//

#ifndef GNN_ACCELERATION_LANGUAGE_CUDA_H
#define GNN_ACCELERATION_LANGUAGE_CUDA_H

#include <unordered_set>
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

    std::string coarsenedKernelCall(ComputeNode* cNode, int cFact, int prevLayer, int weighted = true)
    {
        std::string res = "";
        // Add check for the next coarsening
        // Top layer just divide
        if (prevLayer == -1)
        {
            if (cFact != 0){
                res += "  if ((int)dcols / " + std::to_string(32 * (cFact)) + ") {\n";
            } else {
                res += "  if ((int)dcols) {\n";
            }
        } else
        {
            // Sub-layers - modulo division by the higher layer. Then, see if any remaining
            res += "  if ((dcols % " + std::to_string(32 * (cFact + 1)) + " ) > "
            + std::to_string(32 * (cFact)) + ") {\n";
        }

        res += "    cudaStreamCreate(&stream" + std::to_string(cFact) + ");\n";

        if (prevLayer == -1)
        {
            if (cFact != 0){
                res += "    dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / "
                + std::to_string(32 * (cFact)) + ");\n\
        dim3 blockDim(32, 8);\n";
            } else {
                res += "    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);\n\
        dim3 blockDim((int)dcols, 8);\n";
            }
        } else
        {
            res += "    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);\n";
            if (cFact != 0)
            {
                res += "    dim3 blockDim(32, 8);\n";
            } else
            {
                res += "    dim3 blockDim(dcols %" + std::to_string(32 * (cFact + 1)) + ", 8);\n";
            }
        }

        std::string weightedStr = weighted ? "&val_ptr[start_vals], " : "";

        if (prevLayer == -1)
        {
            if (cFact != 0)
            {
                res += "    " + getKernelName(cNode) + "_kernel" + std::to_string(cFact - 1) + "<<<gridDim, blockDim, 0, stream"
                + std::to_string(cFact) + ">>>(\n\
        oden_array, &offset_ptr[i1 * (nrows + 1)]," + (weightedStr) + " iden_ptr, &col_ptr[start_vals], nrows, dcols);\n";
            } else
            {

                res += "    " + getKernelName(cNode) + "_kernel0<<<gridDim, blockDim, 0, stream0>>>(\n\
        oden_array, &offset_ptr[i1 * (nrows + 1)]," + (weightedStr) + " iden_ptr, &col_ptr[start_vals], nrows, dcols);\n";
            }
        } else
        {
            if (cFact != 0)
            {
                res += "    " + getKernelName(cNode) + "_kernel" + std::to_string(cFact - 1) + "_offset<<<gridDim, blockDim, 0, stream"
                + std::to_string(cFact) + ">>>(\n\
        oden_array, &offset_ptr[i1 * (nrows + 1)]," + (weightedStr) +" iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /"
                + std::to_string(32 * (cFact + 1)) + ") * " + std::to_string(32 * (cFact + 1)) + ");\n";
            } else
            {
                res += "    " + getKernelName(cNode) + "_kernel0_offset<<<gridDim, blockDim, 0, stream0>>>(\n\
        oden_array, &offset_ptr[i1 * (nrows + 1)]," + (weightedStr) + " iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /"
                + std::to_string(32 * (cFact + 1)) + ") * " + std::to_string(32 * (cFact + 1)) + ");\n";
            }
        }
        // Remainder
        if (cFact != 0)
        {
            res += coarsenedKernelCall(cNode, cFact - 1, cFact, weighted);
        }
        // else if (prevLayer == -1)
        // {
        //     res += coarsenedKernelCall(cNode, cFact, cFact, weighted);
        // }
        res += "  }\n";

        // This should be the path if no computation was done earlier
        if (cFact != 0 && prevLayer == -1)
        {
            res += "else {\n";
            res += coarsenedKernelCall(cNode, cFact - 1, -1, weighted);
            res += "}\n";
        }
        return res;
    }

    void generateCudaCodeForCNode(ComputeNode* cNode)
    {
        if (cNode->getOpType() == AGGREGATE_NODE)
        {
            // Get the input
            auto graphInput = cNode->getInput(1);
            auto graphInfo =  graphInput->getDataInfo();

            // Unweighted (Col tile or undirected is not necessary at the moment)
            bool isWeighted = false;
            if (graphInfo->getWeighted())
            {
                isWeighted = true;
            }

            int maxCoarsening = 1;
            for (auto opt: *cNode->getOpts())
            {
                if (opt.first == COARSE_COPT)
                {
                    maxCoarsening = (int)opt.second;
                }
            }

            bool isColTile = hasDOpt(cNode->getInput(1), COL_TILE_DOPT);

            std::string kernelCodeStr = "";

            // TODO eventually change the 32, 8 sizes based on the configurations of the CIR
            //  The 64 here needs to be changed into something else if the blocksize.y is changed
            // TODO add the semiring selection here
            for (int cFact = 0; cFact < maxCoarsening; cFact++)
            {
                kernelCodeStr += "extern \"C\" __global__ void __launch_bounds__(256)\n"
                + getKernelName(cNode) + "_kernel" + std::to_string(cFact) + "(float *__restrict__ C,\n\
                    int *__restrict__ J_indptr_data,\n";

                if (isWeighted)
                {
                    kernelCodeStr += "                                float *__restrict__ A, ";
                }

                kernelCodeStr += "float *__restrict__ B,\n\
                    int *__restrict__ J_indices_data, int nrows,\n\
                    int dcols) {\n\
if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {\n";

                // The local register storage
                for (int j = 0; j <= cFact; j++)
                {
                    kernelCodeStr += "    float local" + std::to_string(j) + " = C[(((((((int)blockIdx.x) * 8)\
+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * " + std::to_string(32 * (cFact + 1)) + ")) + ((int)threadIdx.x)) + " + std::to_string(32 * j) + ")];\n";
                }

                            kernelCodeStr += "\
        for (int j = 0;\n\
             j <\n\
             (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -\n\
              J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
             ++j) {\n";
                            for (int j = 0; j <= cFact; j++)
                            {
                                kernelCodeStr += "\
            local" + std::to_string(j) + " = local" + std::to_string(j) + " +";
                                
                                if (isWeighted)
                                {
                                    kernelCodeStr += "A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +\n\
                                    ((int)threadIdx.y))])] * ";
                                }
                                kernelCodeStr += "(B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + \n\
                                                         ((int)threadIdx.y))])] * \n\
                      dcols) + (((int)blockIdx.y) * " + std::to_string(32 * (cFact + 1)) + ")) + ((int)threadIdx.x) + " + std::to_string(32 * j) + ")]);\n";
                            }

                kernelCodeStr += "             }\n";
                for (int j = 0; j <= cFact; j++)
                {
                kernelCodeStr += "\n\
C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +\n\
(((int)blockIdx.y) * " + std::to_string(32 * (cFact + 1)) + ")) +\n\
((int)threadIdx.x) + " + std::to_string(32 * j) + ")] = local" + std::to_string(j) + ";\n";
                }
                kernelCodeStr += "   }\n}\n\n";
            }

            // Offsets
            for (int cFact = 0; cFact < maxCoarsening - 1; cFact++)
            {
                kernelCodeStr += "extern \"C\" __global__ void __launch_bounds__(256)\n"
                + getKernelName(cNode) + "_kernel" + std::to_string(cFact) + "_offset(float *__restrict__ C,\n\
                    int *__restrict__ J_indptr_data,\n";

                if (isWeighted)
                {
                    kernelCodeStr += "                                float *__restrict__ A, ";
                }

                kernelCodeStr += "float *__restrict__ B,\n\
                    int *__restrict__ J_indices_data, int nrows,\n\
                    int dcols, int offset) {\n\
if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {\n";

                // The local register storage
                for (int j = 0; j <= cFact; j++)
                {
                    kernelCodeStr += "    float local" + std::to_string(j) + " = C[(((((((int)blockIdx.x) * 8)\
+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * " + std::to_string(32 * (cFact + 1)) + ")) + ((int)threadIdx.x)) + "
                    + std::to_string(32 * j) + ") + offset];\n";
                }

                kernelCodeStr += "\
for (int j = 0;\n\
 j <\n\
 (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -\n\
  J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
 ++j) {\n";
                for (int j = 0; j <= cFact; j++)
                {
                    kernelCodeStr += "\
local" + std::to_string(j) + " = local" + std::to_string(j) + " +";

                    if (isWeighted)
                    {
                        kernelCodeStr += "A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +\n\
                                    ((int)threadIdx.y))])] * ";
                    }
                    kernelCodeStr += "(B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + \n\
                                             ((int)threadIdx.y))])] * \n\
          dcols) + (((int)blockIdx.y) * " + std::to_string(32 * (cFact + 1)) + ")) + ((int)threadIdx.x) + "
                    + std::to_string(32 * j) + ") + offset]);\n";
                }

                kernelCodeStr += "             }\n";
                for (int j = 0; j <= cFact; j++)
                {
                kernelCodeStr += "\n\
C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +\n\
(((int)blockIdx.y) * " + std::to_string(32 * (cFact + 1)) + ")) +\n\
((int)threadIdx.x) + " + std::to_string(32 * j) + ") + offset] = local" + std::to_string(j) + ";\n";
                }
                kernelCodeStr += "   }\n}\n\n";
            }

            kernelCode.addCode(kernelCodeStr);

            // This is the kernel call
            std::string aggrKernelCall = ""
            "torch::Tensor " + getKernelName(cNode) + "_call(torch::Tensor input_dense,\n\
                   torch::Tensor offset_graph,\n\
                   torch::Tensor columns_graph,\n\
                   torch::Tensor value_graph\n";
            if (isColTile)
            {
                aggrKernelCall += ", torch::Tensor bounds,\n int segments";
            }
            aggrKernelCall += ") {\n\
auto nrows = global_nrows;\n\
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
float *val_ptr = value_graph.data_ptr<float>();\n";

            if (isColTile)
            {
                aggrKernelCall += "int *bounds_ptr = bounds.data_ptr<int>();\n\
for (int i = 0; i < segments; i++) {\n\
  int i1 = i;\n\
  int start_vals = bounds_ptr[i1 * 2];";
            } else
            {
                aggrKernelCall += "int i1 = 0;\n\
int start_vals = 0;";
            }

            aggrKernelCall += "cudaStream_t ";
            for (int cFact = 0; cFact < maxCoarsening + 1; cFact++)
            {
                aggrKernelCall += "stream" + std::to_string(cFact);
                if (cFact < maxCoarsening)
                {
                    aggrKernelCall += ", ";
                }
            }
            aggrKernelCall += ";\n";

            aggrKernelCall += coarsenedKernelCall(cNode, maxCoarsening, -1, isWeighted);
            if (isColTile)
            {
                aggrKernelCall += "}";
            }
            aggrKernelCall += "return output_dense;\n\
}";
            // Adding the kernel call and setting the name
            kernelCallCode.addCode(aggrKernelCall);
            cNode->setKernelName("gather_forward");
        } else if (cNode->getOp() == NON_LNR_OP_SOFTMAX) {
            std::string kernelCodeStr = "extern \"C\" __global__ void __launch_bounds__(256)\n\
    default_function_kernel_spmm_backward_sddmm_32_nln(\n\
        float *__restrict__ C, // Output dense\n\
        int *__restrict__ J_indptr_data,\n\
        float *__restrict__ A, // Input values\n\
        int *__restrict__ J_indices_data, int nrows) {\n\
  if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < nrows) {\n\
    float local_C = 1e-12;\n\
    for (int j = 0;\n\
         j <\n\
         (J_indptr_data[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) + 1)] -\n\
          J_indptr_data[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);\n\
         ++j) {\n\
      local_C = (local_C + (A[(j + J_indptr_data[((((int)blockIdx.x) * 32) +\n\
                                                  ((int)threadIdx.x))])]));\n\
    }\n\
    C[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] =\n\
        C[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + local_C;\n\
  }\n\
}\n";
            kernelCode.addCode(kernelCodeStr);

            std::string kernelCallCodeStr = "torch::Tensor node_spmv_backward_of_sddmm_nln(torch::Tensor offset_graph,\n\
                                          torch::Tensor columns_graph,\n\
                                          torch::Tensor value_graph,\n\
                                          torch::Tensor bounds, int nrows,\n\
                                          int segments, bool is_directed) {\n\
  // Output\n\
  auto options = torch::TensorOptions()\n\
                     .dtype(torch::kFloat)\n\
                     .requires_grad(true)\n\
                     .device(torch::kCUDA, 0);\n\
  auto output_dense = torch::zeros({nrows, 1}, options);\n\
  float *oden_array = output_dense.data_ptr<float>();\n\
\n\
  // Sparse\n\
  int *offset_ptr = offset_graph.data_ptr<int>();\n\
  int *col_ptr = columns_graph.data_ptr<int>();\n\
  float *val_ptr = value_graph.data_ptr<float>();\n\
  int *bounds_ptr = bounds.data_ptr<int>();\n\
\n\
  for (int i = 0; i < segments; i++) {\n\
    int i1 = i;\n\
    int start_vals = bounds_ptr[i1 * 2];\n\
\n\
    cudaStream_t stream1;\n\
\n\
    cudaStreamCreate(&stream1);\n\
    dim3 gridDim_rem(((int)(nrows - 1) / 32) + 1);\n\
    dim3 blockDim_rem(32);\n\
    default_function_kernel_spmm_backward_sddmm_32_nln<<<gridDim_rem, blockDim_rem,\n\
                                                     0, stream1>>>(\n\
        oden_array, &offset_ptr[i1 * (nrows + 1)], &val_ptr[start_vals],\n\
        &col_ptr[start_vals], nrows);\n\
  }\n\
\n\
  return output_dense;\n\
}";
            kernelCallCode.addCode(kernelCallCodeStr);

        
        } else if (cNode->getOp() == AGGREGATE_EDGE_MUL_SUM_OP) {
            std::string kernelCodeStr = "extern \"C\" __global__ void __launch_bounds__(256)\n\
    default_function_kernel_spmm_backward_sddmm_32_eaggr(\n\
        float *__restrict__ C, // Output dense\n\
        int *__restrict__ J_indptr_data,\n\
        float *__restrict__ A, // Input values\n\
        int *__restrict__ J_indices_data, int nrows) {\n\
  if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < nrows) {\n\
    float local_C = 1e-12;\n\
    for (int j = 0;\n\
         j <\n\
         (J_indptr_data[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) + 1)] -\n\
          J_indptr_data[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);\n\
         ++j) {\n\
      local_C = (local_C + (A[(j + J_indptr_data[((((int)blockIdx.x) * 32) +\n\
                                                  ((int)threadIdx.x))])]));\n\
    }\n\
    C[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] =\n\
        C[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + local_C;\n\
  }\n\
}\n";
            kernelCode.addCode(kernelCodeStr);

            std::string kernelCallCodeStr = "torch::Tensor node_spmv_backward_of_sddmm_eaggr(torch::Tensor offset_graph,\n\
                                          torch::Tensor columns_graph,\n\
                                          torch::Tensor value_graph,\n\
                                          torch::Tensor bounds, int nrows,\n\
                                          int segments, bool is_directed) {\n\
  // Output\n\
  auto options = torch::TensorOptions()\n\
                     .dtype(torch::kFloat)\n\
                     .requires_grad(true)\n\
                     .device(torch::kCUDA, 0);\n\
  auto output_dense = torch::zeros({nrows, 1}, options);\n\
  float *oden_array = output_dense.data_ptr<float>();\n\
\n\
  // Sparse\n\
  int *offset_ptr = offset_graph.data_ptr<int>();\n\
  int *col_ptr = columns_graph.data_ptr<int>();\n\
  float *val_ptr = value_graph.data_ptr<float>();\n\
  int *bounds_ptr = bounds.data_ptr<int>();\n\
\n\
  for (int i = 0; i < segments; i++) {\n\
    int i1 = i;\n\
    int start_vals = bounds_ptr[i1 * 2];\n\
\n\
    cudaStream_t stream1;\n\
\n\
    cudaStreamCreate(&stream1);\n\
    dim3 gridDim_rem(((int)(nrows - 1) / 32) + 1);\n\
    dim3 blockDim_rem(32);\n\
    default_function_kernel_spmm_backward_sddmm_32_eaggr<<<gridDim_rem, blockDim_rem,\n\
                                                     0, stream1>>>(\n\
        oden_array, &offset_ptr[i1 * (nrows + 1)], &val_ptr[start_vals],\n\
        &col_ptr[start_vals], nrows);\n\
  }\n\
\n\
  return output_dense;\n\
}";
            kernelCallCode.addCode(kernelCallCodeStr);

        }
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

        std::unordered_set<std::string> encountedOps;
        for (int i = 0; i < program.size(); i++)
        {
            CIRNode* outNode = program[i];
            auto oNode = dynamic_cast<ComputeNode*>(outNode);
            if (oNode)
            {
                auto cNode = dynamic_cast<ComputeNode*>(outNode);
                std::string kernelName = getKernelName(cNode);

                if (encountedOps.find(kernelName) == encountedOps.end())
                {
                    generateCudaCodeForCNode(cNode);
                    encountedOps.insert(kernelName);
                }
            } else {
                auto loopNode = dynamic_cast<TrainingLoopNode*>(outNode);
                for (int ix = 0; ix < loopNode->getLoopNodeNum(); ix++)
                {
                    CIRNode* inNode = loopNode->getNode(ix);
                    auto cNode = dynamic_cast<ComputeNode*>(inNode);
                    std::string kernelName = getKernelName(cNode);
                    if (encountedOps.find(kernelName) == encountedOps.end())
                    {
                        generateCudaCodeForCNode(cNode);
                        encountedOps.insert(kernelName);
                    }
                }
            }
        }
    }

    void generateCudaTransferCodeForUniqueInput(ComputeNode* cNode,
        std::unordered_set<std::string> &encounteredStrings)
    {
        std::string inputTransferCode = "";
        // TODO need to the same for the backward pass' data
        // Add BOTH precode and postcode
        for (int inpI = 0; inpI < cNode->getNumInputs(); inpI++)
        {
            auto inputData = cNode->getInput(inpI);
            // Check if the string has been encountered before
            if (encounteredStrings.find(inputData->getName()) == encounteredStrings.end()) {
                // For now only generate the transfer code for CSR type graphs
                auto inputInfo =  inputData->getDataInfo();
                if (inputInfo->getFormat() == CSR_STYPE)
                {
                    int indexData = (int)encounteredStrings.size();
                    if (inputInfo->getIndex() != -1)
                    {
                        indexData = inputInfo->getIndex();
                    }
                    encounteredStrings.insert(inputData->getName());

                    inputInfo->setIndex(indexData);

                    bool isColTile = hasDOpt(inputData, COL_TILE_DOPT);
                    inputTransferCode += "  int *dA_csrOffsets"+std::to_string(indexData)+", *dA_columns"+std::to_string(indexData)+"; \n\
  float *dA_values"+std::to_string(indexData)+";\n\
\n\
  CUDA_CHECK(cudaMalloc((void **)&dA_columns"+std::to_string(indexData)+", nvals"+std::to_string(indexData)+" * sizeof(int)));\n\
  CUDA_CHECK(cudaMalloc((void **)&dA_values"+std::to_string(indexData)+", nvals"+std::to_string(indexData)+" * sizeof(float)));\n";
                    if (isColTile)
                    {
                        inputTransferCode += "\n\
  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets"+std::to_string(indexData)+", (nrows + 1) * segments_" + inputData->getName() + " * sizeof(int)));\n\
\n\
  CUDA_CHECK(cudaMemcpy(dA_csrOffsets"+std::to_string(indexData)+", offset_ptr_" + inputData->getName() + ",\n\
                        (nrows + 1) * segments_" + inputData->getName() + " * sizeof(int), cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dA_columns"+std::to_string(indexData)+", col_ptr_" + inputData->getName() + ", nvals"+std::to_string(indexData)+" * sizeof(int),\n\
                        cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dA_values"+std::to_string(indexData)+", val_ptr_" + inputData->getName() + ", nvals"+std::to_string(indexData)+" * sizeof(float),\n\
                        cudaMemcpyHostToDevice));\n\
  torch::Tensor t_offsets"+std::to_string(indexData)+" =\n\
      torch::from_blob(dA_csrOffsets"+std::to_string(indexData)+", {(nrows+ 1) * segments_" + inputData->getName() + "}, options_cu_int);\n";
                    } else
                    {
                        inputTransferCode += "  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets"+std::to_string(indexData)+", (nrows + 1) * sizeof(int)));\n\
\n\
  CUDA_CHECK(cudaMemcpy(dA_csrOffsets"+std::to_string(indexData)+", adj"+std::to_string(indexData)+".offset_ptr(),\n\
                        (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dA_columns"+std::to_string(indexData)+", adj"+std::to_string(indexData)+".ids_ptr(), nvals"+std::to_string(indexData)+" * sizeof(int),\n\
                        cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dA_values"+std::to_string(indexData)+", adj"+std::to_string(indexData)+".vals_ptr(), nvals"+std::to_string(indexData)+" * sizeof(float),\n\
                        cudaMemcpyHostToDevice));\n\
  torch::Tensor t_offsets"+std::to_string(indexData)+" =\n\
      torch::from_blob(dA_csrOffsets"+std::to_string(indexData)+", {nrows+ 1}, options_cu_int);\n";
                    }

                    inputTransferCode += "  torch::Tensor t_cols"+std::to_string(indexData)+" = torch::from_blob(dA_columns"+std::to_string(indexData)+", {nvals"+std::to_string(indexData)+"}, options_cu_int);\n\
\n\
  torch::Tensor t_vals"+std::to_string(indexData)+" =\n\
      torch::from_blob(dA_values"+std::to_string(indexData)+", {nvals"+std::to_string(indexData)+"}, options_cu_float_ngrad);\n";

                    inputTransferCode += "  global_offset_graph.push_back(t_offsets"+std::to_string(indexData)+");\n\
  global_columns_graph.push_back(t_cols"+std::to_string(indexData)+");\n\
  global_value_graph.push_back(t_vals"+std::to_string(indexData)+");\n";

                    // These are graphs for backprop
                    if (!inputInfo->getDirected())
                    {
                        inputTransferCode += "  global_offset_graph.push_back(t_offsets"+std::to_string(indexData)+");\n\
    global_columns_graph.push_back(t_cols"+std::to_string(indexData)+");\n\
    global_value_graph.push_back(t_vals"+std::to_string(indexData)+");\n";
                    } else
                    {
                         inputTransferCode += "  int *dA_csrOffsets"+std::to_string(indexData)+"_b, *dA_columns"+std::to_string(indexData)+"_b; \n\
  float *dA_values"+std::to_string(indexData)+"_b;\n\
\n\
  CUDA_CHECK(cudaMalloc((void **)&dA_columns"+std::to_string(indexData)+"_b, nvals"+std::to_string(indexData)+" * sizeof(int)));\n\
  CUDA_CHECK(cudaMalloc((void **)&dA_values"+std::to_string(indexData)+"_b, nvals"+std::to_string(indexData)+" * sizeof(float)));\n";
                    if (isColTile)
                    {
                        inputTransferCode += "\n\
  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets"+std::to_string(indexData)+"_b, (nrows + 1) * segments_" + inputData->getName() + "_b * sizeof(int)));\n\
\n\
  CUDA_CHECK(cudaMemcpy(dA_csrOffsets"+std::to_string(indexData)+"_b, offset_ptr_" + inputData->getName() + "_b,\n\
                        (nrows + 1) * segments_" + inputData->getName() + "_b * sizeof(int), cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dA_columns"+std::to_string(indexData)+"_b, col_ptr_" + inputData->getName() + "_b, nvals"+std::to_string(indexData)+" * sizeof(int),\n\
                        cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dA_values"+std::to_string(indexData)+"_b, val_ptr_" + inputData->getName() + "_b, nvals"+std::to_string(indexData)+" * sizeof(float),\n\
                        cudaMemcpyHostToDevice));\n\
  torch::Tensor t_offsets"+std::to_string(indexData)+"_b =\n\
      torch::from_blob(dA_csrOffsets"+std::to_string(indexData)+"_b, {(nrows+ 1) * segments_" + inputData->getName() + "_b}, options_cu_int);\n";
                    } else
                    {
                        inputTransferCode += "  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets"+std::to_string(indexData)+"_b, (nrows + 1) * sizeof(int)));\n\
\n\
  CUDA_CHECK(cudaMemcpy(dA_csrOffsets"+std::to_string(indexData)+"_b, adj"+std::to_string(indexData)+"_b.offset_ptr(),\n\
                        (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dA_columns"+std::to_string(indexData)+"_b, adj"+std::to_string(indexData)+"_b.ids_ptr(), nvals"+std::to_string(indexData)+" * sizeof(int),\n\
                        cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dA_values"+std::to_string(indexData)+"_b, adj"+std::to_string(indexData)+"_b.vals_ptr(), nvals"+std::to_string(indexData)+" * sizeof(float),\n\
                        cudaMemcpyHostToDevice));\n\
  torch::Tensor t_offsets"+std::to_string(indexData)+"_b =\n\
      torch::from_blob(dA_csrOffsets"+std::to_string(indexData)+"_b, {nrows+ 1}, options_cu_int);\n";
                    }

                    inputTransferCode += "  torch::Tensor t_cols"+std::to_string(indexData)+"_b = torch::from_blob(dA_columns"+std::to_string(indexData)+"_b, {nvals"+std::to_string(indexData)+"}, options_cu_int);\n\
\n\
  torch::Tensor t_vals"+std::to_string(indexData)+"_b =\n\
      torch::from_blob(dA_values"+std::to_string(indexData)+"_b, {nvals"+std::to_string(indexData)+"}, options_cu_float_ngrad);\n";

                    inputTransferCode += "  global_offset_graph.push_back(t_offsets"+std::to_string(indexData)+"_b);\n\
  global_columns_graph.push_back(t_cols"+std::to_string(indexData)+"_b);\n\
  global_value_graph.push_back(t_vals"+std::to_string(indexData)+"_b);\n";
                    }
                }
            }
        }
        preCode.addCode(inputTransferCode);
    }

    // You don't need transfer operations if it is produced by some computation operation
    void cudaTransfer(std::vector<CIRNode*>& program)
    {
        std::unordered_set<std::string> encounteredStrings;
        // Iterate through the entire program
        for (int i = 0; i < program.size(); i++)
        {
            CIRNode* outNode = program[i];
            auto oNode = dynamic_cast<ComputeNode*>(outNode);
            if (oNode)
            {
                generateCudaTransferCodeForUniqueInput(oNode, encounteredStrings);
            } else
            {
                auto loopNode = dynamic_cast<TrainingLoopNode*>(outNode);
                for (int ix = 0; ix < loopNode->getLoopNodeNum(); ix++)
                {
                    CIRNode* inNode = loopNode->getNode(ix);
                    auto cNode = dynamic_cast<ComputeNode*>(inNode);
                    generateCudaTransferCodeForUniqueInput(cNode, encounteredStrings);
                }
            }
        }
    }

    void dataPrep(std::vector<CIRNode*>& program) override
    {
        // TODO make the transfer based on the data and the transformations applied
        // Add the graph parts to a vector
        std::string torchTypesStr = "  torch::Device device(torch::kCUDA);\n\
  auto options_cu_int = torch::TensorOptions()\n\
                            .dtype(torch::kInt)\n\
                            .requires_grad(false)\n\
                            .device(torch::kCUDA, 0);\n\
  auto options_cu_float_grad = torch::TensorOptions()\n\
                                   .dtype(torch::kFloat)\n\
                                   .requires_grad(true)\n\
                                   .device(torch::kCUDA, 0);\n\
  auto options_cu_float_ngrad = torch::TensorOptions()\n\
                                    .dtype(torch::kFloat)\n\
                                    .requires_grad(false)\n\
                                    .device(torch::kCUDA, 0);\n\
  auto options_cu_bool = torch::TensorOptions()\n\
                             .dtype(torch::kBool)\n\
                             .requires_grad(false)\n\
                             .device(torch::kCUDA, 0);\n\
  auto options_cu_long =\n\
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA, 0);\n";
        preCode.addCode(torchTypesStr);

        // For now there's no slicing for a dense input so just use a hard-coded code-generation.
        std::string labelMaskStr = "int *dL;\n\
  float *dB;\n\
  bool *d_train_mask, *d_valid_mask, *d_test_mask;\n\
\n\
  CUDA_CHECK(cudaMalloc((void **)&dB, (nrows * emb_size) * sizeof(float)));\n\
  CUDA_CHECK(cudaMalloc((void **)&dL, nrows * sizeof(long)));\n\
  CUDA_CHECK(cudaMalloc((void **)&d_train_mask, nrows * sizeof(bool)));\n\
  CUDA_CHECK(cudaMalloc((void **)&d_valid_mask, nrows * sizeof(bool)));\n\
  CUDA_CHECK(cudaMalloc((void **)&d_test_mask, nrows * sizeof(bool)));\n\
\n\
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
\n\
  torch::Tensor t_iden =\n\
      torch::from_blob(dB, {nrows, emb_size}, options_cu_float_grad);\n\
  torch::Tensor t_labs = torch::from_blob(dL, {nrows}, options_cu_long);\n\
\n\
  torch::Tensor t_train_mask =\n\
      torch::from_blob(d_train_mask, {nrows}, options_cu_bool);\n\
  torch::Tensor t_valid_mask =\n\
      torch::from_blob(d_valid_mask, {nrows}, options_cu_bool);\n\
  torch::Tensor t_test_mask =\n\
      torch::from_blob(d_test_mask, {nrows}, options_cu_bool);";
        preCode.addCode(labelMaskStr);
        cudaTransfer(program);

        std::string cleanCuda = " CUDA_CHECK(cudaFree(dB));";
        postCode.addCode(cleanCuda);

    }
};

#endif //GNN_ACCELERATION_LANGUAGE_CUDA_H
