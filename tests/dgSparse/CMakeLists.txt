cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(gala_cuda LANGUAGES CUDA CXX)

find_package(Torch REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "Need OpenMP")
endif ()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} util)
# Link against LibTorch
link_libraries("${TORCH_LIBRARIES}" cudart cusparse)

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