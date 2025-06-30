#ifndef FILE_OP_H
#define FILE_OP_H
#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <torch/torch.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "common.h"

using torch::Tensor;

enum class FileType { mmap, raw, memory, mmap_write, empty };

class FileHandler {
 public:
  FileType mode_;  // 0 for mmap, 1 for read from file
  std::string filename_ = "";
  int64_t filesize_ = 0;
  int fd_ = -1;
  char *mmap_ = nullptr;
  FileHandler(std::string filename, FileType mode = FileType::mmap);
  void openFile();
  void readFile(int64_t size, int64_t offset, void *buf);
  void closeFile();
  void *readAll();
  void writeFile(int64_t size, void *buf);

  template <class T>
  void readAllToVec(std::vector<T> &vec) {
    openFile();
    auto ptr = (T *)readAll();
    closeFile();
    vec.assign(ptr, ptr + filesize_ / sizeof(T));
    delete[] ptr;
  }

  template <class T>
  void fetchData(const std::vector<Index> &index, int64_t feature_len,
                 std::vector<T> &buf) {
    int64_t feature_size = feature_len * sizeof(T);
    if (buf.size() < index.size() * feature_len)
      buf.resize(index.size() * feature_len);
    if (mode_ == FileType::mmap || mode_ == FileType::memory) {
      ASSERT(mmap_ != nullptr);
      for (int i = 0; i < index.size(); ++i) {
        memcpy(buf.data() + i * feature_len, mmap_ + index[i] * feature_size,
               feature_size);
      }
    } else if (mode_ == FileType::raw) {
      for (int i = 0; i < index.size(); ++i) {
        pread(fd_, buf.data() + i * feature_len, feature_size,
              index[i] * feature_size);
      }
    } else if (mode_ == FileType::empty) {
      return;
    }
  }

  template <class T>
  void fetchData(const std::vector<Index> &index, int64_t feature_len,
                 Tensor &buf) {
    int64_t feature_size = feature_len * sizeof(T);
    if (mode_ == FileType::mmap || mode_ == FileType::memory) {
      ASSERT(mmap_ != nullptr);
      for (int i = 0; i < index.size(); ++i) {
        memcpy(buf.data<T>() + i * feature_len, mmap_ + index[i] * feature_size,
               feature_size);
      }
    } else if (mode_ == FileType::raw) {
      for (int i = 0; i < index.size(); ++i) {
        pread(fd_, buf.data<T>() + i * feature_len, feature_size,
              index[i] * feature_size);
      }
    } else if (mode_ == FileType::empty) {
      return;
    }
    timestamp(t1);
  }

  template <class T>
  void fetchAllData(std::vector<T> &buf) {
    if (buf.size() < filesize_) buf.resize(filesize_);
    if (mode_ == FileType::mmap || mode_ == FileType::memory) {
      ASSERT(mmap_ != nullptr);
      memcpy(buf.data(), mmap_, filesize_);
    } else if (mode_ == FileType::raw) {
      pread(fd_, buf.data(), filesize_, 0);
    }
  }

  template <class T>
  void fetchAllData(Tensor &buf) {
    if (mode_ == FileType::mmap || mode_ == FileType::memory) {
      ASSERT(mmap_ != nullptr);
      memcpy(buf.data<T>(), mmap_, filesize_);
    } else if (mode_ == FileType::raw) {
      pread(fd_, buf.data<T>(), filesize_, 0);
    }
  }
};

#endif