cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(gala_cpu LANGUAGES CXX)

# set(CMAKE_CXX_COMPILER icpx) 

find_package(Torch REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "Need OpenMP")
endif ()

link_libraries("${TORCH_LIBRARIES}")

message("Compiler: COMILER_ID is ${CMAKE_CXX_COMPILER_ID}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -DICC")
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

add_compile_definitions(GALA_TORCH)
add_compile_definitions(GN_1)
add_compile_definitions(PT_0)
add_compile_definitions(ST_0)
add_compile_definitions(A_ALLOC)

add_executable(gala npy_gcn_model.cpp)
target_compile_features(gala PRIVATE cxx_std_14)