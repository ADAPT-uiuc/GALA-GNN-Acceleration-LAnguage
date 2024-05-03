cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(gala_gather)

find_package(Torch REQUIRED)

# Define our library target
add_library(gala_gather SHARED gather.cpp)
# Enable C++11
target_compile_features(gala_gather PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(gala_gather "${TORCH_LIBRARIES}")
# Link against OpenCV
#target_link_libraries(gala_gather opencv_core opencv_imgproc)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "Need OpenMP")
endif ()

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