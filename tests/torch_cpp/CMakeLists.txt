cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(gala_cuda LANGUAGES CUDA CXX)

set(CMAKE_CXX_COMPILER icpx) 

find_package(Torch REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "Need OpenMP")
endif ()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories("/home/damitha2/boost/include")
# include_directories("/home/damitha2/numa/libnuma/usr/local/include")
# Link against LibTorch
link_libraries("${TORCH_LIBRARIES}" cudart cusparse)

#target_link_libraries(gala_cuda PUBLIC )

message("Compiler: COMILER_ID is ${CMAKE_CXX_COMPILER_ID}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=0  -march=native -xCORE-AVX512 -O3 -DICC -restrict")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed -lpthread -lm -ldl")
set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} -lnuma")

if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmkl_gnu_thread -lgomp")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgomp")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL IntelLLVM)
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmkl_intel_thread -liomp5")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -liomp5")
endif ()

message("CMAKE_CXX_FLAGS - ${CMAKE_CXX_FLAGS}")
message("CMAKE_EXE_LINKER_FLAGS - ${CMAKE_EXE_LINKER_FLAGS}")
message("CMAKE_CXX_STANDARD_LIBRARIES - ${CMAKE_CXX_STANDARD_LIBRARIES}")

add_compile_options(-Xcompiler -fopenmp -march=native -O3)

# Define our library target
add_compile_definitions(GALA_TORCH)
add_compile_definitions(GN_1)
add_compile_definitions(PT_0)
add_compile_definitions(ST_0)
add_compile_definitions(A_ALLOC)
add_executable(gala_cuda gcn_model.cu)
# Enable C++11
target_compile_features(gala_cuda PRIVATE cxx_std_14)

add_executable(gala_cuda_tile gcn_model_col_tile.cu)
target_compile_features(gala_cuda_tile PRIVATE cxx_std_14)

add_executable(gala_cuda_tir gcn_model_tir_base.cu)
target_compile_features(gala_cuda_tir PRIVATE cxx_std_14)

add_executable(gala_cuda_tile_tir gcn_model_col_tile_tir.cu)
target_compile_features(gala_cuda_tile_tir PRIVATE cxx_std_14)

add_executable(gala_cuda_tile_tir_op gcn_model_col_tile_tir_op.cu)
target_compile_features(gala_cuda_tile_tir_op PRIVATE cxx_std_14)

add_executable(gala_cuda_tile_tir_op_tim gcn_model_col_tile_tir_op_tim.cu)
target_compile_features(gala_cuda_tile_tir_op_tim PRIVATE cxx_std_14)

# dcsr
add_executable(gala_cuda_tir_dcsr gcn_model_tir_dcsr.cu)
target_compile_features(gala_cuda_tir_dcsr PRIVATE cxx_std_14)

# bool
add_executable(gala_cuda_tile_tir_mask gcn_model_col_tile_tir_mask.cu)
target_compile_features(gala_cuda_tile_tir_mask PRIVATE cxx_std_14)

# npy
add_executable(gala npy_gcn_model.cu)
target_compile_features(gala PRIVATE cxx_std_14)
add_executable(gala_tir npy_gcn_model_tir_base.cu)
target_compile_features(gala_tir PRIVATE cxx_std_14)
add_executable(gala_tile_tir npy_gcn_model_col_tile_tir.cu)
target_compile_features(gala_tile_tir PRIVATE cxx_std_14)
add_executable(gala_tile_tir_op npy_gcn_model_col_tile_tir_op.cu)
target_compile_features(gala_tile_tir_op PRIVATE cxx_std_14)
add_executable(gala_tile_tir_op_sparse npy_gcn_model_col_tile_tir_op_sparse.cu)
target_compile_features(gala_tile_tir_op_sparse PRIVATE cxx_std_14)
add_executable(gala_tile_tir_op_tim npy_gcn_model_col_tile_tir_op_tim.cu)
target_compile_features(gala_tile_tir_op_tim PRIVATE cxx_std_14)
add_executable(gala_tile_tir_op_tim2 npy_gcn_model_col_tile_tir_op_tim2.cu)
target_compile_features(gala_tile_tir_op_tim2 PRIVATE cxx_std_14)
add_executable(gala_tile_tir_op_tim2_glob npy_gcn_model_col_tile_tir_op_tim2_glob.cu)
target_compile_features(gala_tile_tir_op_tim2_glob PRIVATE cxx_std_14)
add_executable(gala_tile_tir_op_tim2_glob_sparse npy_gcn_model_col_tile_tir_op_tim2_glob_sparse.cu)
target_compile_features(gala_tile_tir_op_tim2_glob_sparse PRIVATE cxx_std_14)
# target_compile_options(gala_tile_tir_op_tim2_glob PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)

add_executable(gala_tile_tir_op_tim2_ogbn npy_gcn_model_col_tile_tir_op_tim2_ogbn.cu)
target_compile_features(gala_tile_tir_op_tim2_ogbn PRIVATE cxx_std_14)

add_executable(gala_tile_tir_op_cus npy_gcn_model_col_tile_tir_op_cus.cu)
target_compile_features(gala_tile_tir_op_cus PRIVATE cxx_std_14)

add_executable(gala_test npy_gcn_model_test.cu)
target_compile_features(gala_test PRIVATE cxx_std_14)

add_executable(gala_sddvv npy_sddvv_test.cu)
target_compile_features(gala_sddvv PRIVATE cxx_std_14)

add_executable(gala_tir_mask npy_gcn_model_tir_base_masked_graph.cu)
target_compile_features(gala_tir_mask PRIVATE cxx_std_14)

add_executable(gala_tir_sample npy_gcn_model_tir_base_sample.cu)
target_compile_features(gala_tir_sample PRIVATE cxx_std_14)
add_executable(gala_tile_tir_op_sample npy_gcn_model_col_tile_tir_op_sample.cu)
target_compile_features(gala_tile_tir_op_sample PRIVATE cxx_std_14)
add_executable(gala_tile_tir_op_sample_data npy_gcn_model_col_tile_tir_op_sample_data.cu)
target_compile_features(gala_tile_tir_op_sample_data PRIVATE cxx_std_14)

add_executable(gala_gat_tile_tir_op npy_gat_model_col_tile_tir_op.cu)
target_compile_features(gala_gat_tile_tir_op PRIVATE cxx_std_14)
add_executable(gala_gat_tile_tir_op_sparse npy_gat_model_col_tile_tir_op_sparse.cu)
target_compile_features(gala_gat_tile_tir_op_sparse PRIVATE cxx_std_14)

add_executable(gala_gin_tile_tir_op npy_gin_model_col_tile_tir_op.cu)
target_compile_features(gala_gin_tile_tir_op PRIVATE cxx_std_14)
add_executable(gala_gin_tile_tir_op_tim2_glob npy_gin_model_col_tile_tir_op_tim2_glob.cu)
target_compile_features(gala_gin_tile_tir_op_tim2_glob PRIVATE cxx_std_14)

add_executable(gala_sweep_op npy_gcn_model_col_tile_tir_op_sweep.cu)
target_compile_features(gala_sweep_op PRIVATE cxx_std_14)