#ifndef COMMON_H
#define COMMON_H
#include <cuda_runtime.h>
#include <stdint.h>
#include <sys/stat.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>

#include "Yaml.hpp"
#include "spdlog/spdlog.h"

using Index = int64_t;
using EtypeIndex = int64_t;
using std::make_shared;
using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

const torch::TensorOptions float32_option =
    torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);

const torch::TensorOptions float16_option =
    torch::TensorOptions().dtype(torch::kFloat16).requires_grad(false);

const torch::TensorOptions int64_option =
    torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);

const torch::TensorOptions int32_option =
    torch::TensorOptions().dtype(torch::kInt32).requires_grad(false);

static inline int randInt(const int &min, const int &max, uint32_t &x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return (x % (max - min)) + min;
}

static inline double zeroToOne(uint32_t &x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x * 2.3283064365386963e-10;  // 2^-32
}

double cudaMemInfo();  // return MB

void showCpuMem();

void showCpuMemCurrProc();

#define ASSERTWITH(condition, args...) \
  if (unlikely(!(condition))) {        \
    SPDLOG_WARN(args);                 \
    exit(1);                           \
  }

#define ASSERT(condition)          \
  if (unlikely(!(condition))) {    \
    SPDLOG_WARN("ASSERT FAILURE"); \
    exit(1);                       \
  }

inline bool fexist(const std::string &name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();
inline double getDuration(
    std::chrono::time_point<std::chrono::system_clock> a,
    std::chrono::time_point<std::chrono::system_clock> b) {
  return std::chrono::duration<double>(b - a).count();
}

#define checkCudaErrors(status)                             \
  do {                                                      \
    if (status != 0) {                                      \
      fprintf(stderr, "CUDA failure at [%s] (%s:%d): %s\n", \
              __PRETTY_FUNCTION__, __FILE__, __LINE__,      \
              cudaGetErrorString(status));                  \
      cudaDeviceReset();                                    \
      abort();                                              \
    }                                                       \
  } while (0)

double getAverageTimeWithWarmUp(const std::function<void()> &f);

inline double getCUDATime(const std::function<void()> &f) {
  checkCudaErrors(cudaDeviceSynchronize());
  timestamp(t0);
  f();
  checkCudaErrors(cudaDeviceSynchronize());
  timestamp(t1);
  return getDuration(t0, t1);
}

void checkConfig(Yaml::Node &config);

#endif
