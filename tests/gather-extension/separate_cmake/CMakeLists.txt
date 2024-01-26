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