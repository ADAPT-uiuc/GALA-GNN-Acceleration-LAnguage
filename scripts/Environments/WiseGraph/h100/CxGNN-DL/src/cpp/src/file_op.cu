#include "common.h"
#include "file_op.h"
#include "spdlog/spdlog.h"

FileHandler::FileHandler(std::string filename, FileType mode)
    : filename_(filename), mode_(mode) {}

void FileHandler::openFile() {
  struct stat file_info = {0};
  if (mode_ == FileType::mmap) {
    fd_ = open(filename_.c_str(), O_RDONLY, (mode_t)0600);
    ASSERTWITH(fd_ != -1, "error open file, path {}", filename_);
    ASSERTWITH(fstat(fd_, &file_info) != -1, "error get file size, path {}",
               filename_);
    ASSERTWITH(file_info.st_size != 0, "empty file, path {}", filename_);
    filesize_ = file_info.st_size;
    mmap_ = (char *)mmap(0, file_info.st_size, PROT_READ, MAP_SHARED, fd_, 0);
    ASSERTWITH(mmap_ != MAP_FAILED, "fail mmap, path {}", filename_);
    // ASSERTWITH(0 == madvise(mmap_, (intmax_t)file_info.st_size, MADV_RANDOM),
    // "fail madvise {}",
    //  filename_);
  } else if (mode_ == FileType::raw) {
    int flags = O_RDONLY;
    fd_ = open(filename_.c_str(), flags);
    if (fstat(fd_, &file_info) == -1) {
      perror("Error getting the file size");
      exit(EXIT_FAILURE);
    }
    if (file_info.st_size == 0) {
      fprintf(stderr, "Error: File is empty, nothing to do\n");
      exit(EXIT_FAILURE);
    }
    filesize_ = file_info.st_size;
  } else if (mode_ == FileType::memory) {
    fd_ = open(filename_.c_str(), O_RDONLY, (mode_t)0600);
    ASSERTWITH(fd_ != -1, "error open file, path {}", filename_);
    ASSERTWITH(fstat(fd_, &file_info) != -1, "error get file size, path {}",
               filename_);
    ASSERTWITH(file_info.st_size != 0, "empty file, path {}", filename_);
    filesize_ = file_info.st_size;
    mmap_ = (char *)mmap(0, file_info.st_size, PROT_READ, MAP_SHARED, fd_, 0);
    ASSERTWITH(mmap_ != MAP_FAILED, "fail mmap, path {}", filename_);

    char *mem = new char[filesize_];
    memcpy(mem, mmap_, filesize_ * sizeof(char));
    munmap(mmap_, filesize_);
    mmap_ = mem;
    close(fd_);
  } else if (mode_ == FileType::empty) {
    return;
  } else
    abort();
  SPDLOG_DEBUG("file size {}", filesize_);
}

void FileHandler::readFile(int64_t size, int64_t offset, void *buf) {
  if (offset + size >= filesize_) {
    SPDLOG_WARN("read out of bound");
    abort();
  }
  if (mode_ == FileType::mmap) {
    assert(mmap_ != nullptr);
    memcpy(buf, mmap_ + offset, size);
  } else if (mode_ == FileType::raw) {
    ssize_t result = pread(fd_, buf, size, offset);
    if (result < 0) {
      exit(EXIT_FAILURE);
    }
  } else if (mode_ == FileType::empty) {
    return;
  } else
    abort();
}

void FileHandler::writeFile(int64_t size, void *buf) {
  ASSERT(mode_ == FileType::mmap_write);
  fd_ = open(filename_.c_str(), O_RDWR | O_CREAT, (mode_t)0600);
  ASSERTWITH(fd_ != -1, "error open file, path {}", filename_);
  filesize_ = size + 1;
  lseek(fd_, size, 1);
  write(fd_, "", 1);
  mmap_ =
      (char *)mmap(0, filesize_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  ASSERTWITH(mmap_ != MAP_FAILED, "fail mmap, path {}", filename_);
  memcpy(mmap_, buf, size);
  msync(mmap_, filesize_, MS_SYNC);
}

void FileHandler::closeFile() {
  if (mode_ == FileType::empty) {
    return;
  }
  if (mode_ == FileType::mmap || mode_ == FileType::mmap_write) {
    munmap(mmap_, filesize_);
  }
  close(fd_);
}

void *FileHandler::readAll() {
  void *buf = new char[filesize_];
  if (mode_ == FileType::mmap) {
    memcpy(buf, mmap_, filesize_);
  } else if (mode_ == FileType::raw) {
    ssize_t result = pread(fd_, buf, filesize_, 0);
    if (result < 0) {
      exit(EXIT_FAILURE);
    }
  } else
    abort();
  return buf;
}