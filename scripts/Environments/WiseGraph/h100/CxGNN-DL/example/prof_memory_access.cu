#include <omp.h>
#include <stdint.h>

#include <chrono>
#include <iostream>
#include <random>
using namespace std;

static inline int randInt(const int &min, const int &max, uint32_t &x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return (x % (max - min)) + min;
}

#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();
inline double getDuration(
    std::chrono::time_point<std::chrono::system_clock> a,
    std::chrono::time_point<std::chrono::system_clock> b) {
  return std::chrono::duration<double>(b - a).count();
}

int main() {
  int n = 1000000;
  int feat = 512;
  float *a = new float[feat * n];
  int num_to_fetch = 1000;
  vector<int> index(num_to_fetch);
  uint32_t seed = 2313;
  for (int i = 0; i < num_to_fetch; ++i) {
    index[i] = randInt(0, n, seed);
  }
  vector<float> b(num_to_fetch * feat);

  float *c;
  cudaMalloc(&c, sizeof(float) * num_to_fetch * feat);

  int num_thread = 32;
  vector<int *> indexes(num_thread);
  for (int i = 0; i < num_thread; ++i) {
    indexes[i] = new int[num_to_fetch];
    for (int j = 0; j < num_to_fetch; ++j) {
      indexes[i][j] = randInt(0, n, seed);
    }
  }
  vector<float *> bs(num_thread);
  for (int i = 0; i < num_thread; ++i) {
    bs[i] = new float[num_to_fetch * feat];
  }
  vector<float *> cs(num_thread);
  for (int i = 0; i < num_thread; ++i) {
    cudaMalloc(&cs[i], sizeof(float) * num_to_fetch * feat);
  }
  float total = 0;
  for (int trial = 0; trial < 10; ++trial) {
    cudaDeviceSynchronize();
    timestamp(t1);
#pragma omp parallel for
    for (int iter = 0; iter < num_thread; ++iter) {
      for (int i = 0; i < num_to_fetch; ++i) {
        memcpy(bs[iter] + i * feat, a + indexes[iter][i] * feat,
               feat * sizeof(float));
      }
      cudaMemcpy(cs[iter], bs[iter], sizeof(float) * num_to_fetch * feat,
                 cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    timestamp(t2);
    if (trial > 0) total += getDuration(t1, t2);
    if (trial == 9) {
      cout << "bandwidth: "
           << (sizeof(float) * num_to_fetch * feat) * num_thread / 1e9 / total *
                  9
           << endl;
    }
  }
  //   for (int iter = 0; iter < 10; ++iter) {
  //     cudaDeviceSynchronize();
  //     timestamp(t1);
  // #pragma omp parallel for
  //     for (int i = 0; i < num_to_fetch; ++i) {
  //       // cout << omp_get_num_threads() << endl;
  //       memcpy(b.data() + i * feat, a + index[i] * feat, feat *
  //       sizeof(float));
  //     }
  //     cudaMemcpy(c, b.data(), sizeof(float) * num_to_fetch * feat,
  //                cudaMemcpyHostToDevice);
  //     cudaDeviceSynchronize();
  //     timestamp(t2);
  //     if (iter == 8) {
  //       cout << "memcpy time: " << getDuration(t1, t2) << endl;
  //       cout << "bandwidth: "
  //            << (sizeof(float) * num_to_fetch * feat) / 1e9 / getDuration(t1,
  //            t2)
  //            << endl;
  //     }
  //   }
}