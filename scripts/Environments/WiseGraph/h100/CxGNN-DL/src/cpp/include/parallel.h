#ifndef __PARALLEL_H__
#define __PARALLEL_H__
#include <pthread.h>
#include <sched.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "queue.h"
#include "spdlog/spdlog.h"
#include "transfer_worker.h"
#include "worker.h"
using std::atomic;
using std::condition_variable;
using std::mutex;
using std::thread;

template <typename T>
class Parallelizer : public Worker<T> {
  // Interestingly, a parallelizer is also a worker
  // Concretely, it is a group of workers which can be used as one worker
 public:
  Parallelizer(int num_threads, int max_in_flight,
               const std::vector<Worker<T> *> &worker_vec, int bind_method,
               int num_threads_base = 0)
      : num_threads_(num_threads),
        max_in_flight_(max_in_flight),
        worker_vec_(worker_vec),
        bind_method_(bind_method),
        num_threads_base_(num_threads_base) {
    max_lock_ = new std::mutex();
    get_lock_ = new std::mutex();
    repeat_get_lock_ = new std::mutex();
    finished_ = true;
    max_cv_ = new std::condition_variable();
    in_flight_ = 0;
    paused_vec_ = new atomic<bool>[num_threads];
    thread_vec_ = new thread *[num_threads];
    for (int i = 0; i < num_threads; ++i) {
      paused_vec_[i] = true;
      thread_vec_[i] = nullptr;
    }
    curr_repeat = 0;
  }

  ~Parallelizer() {
    stop();
    delete[] paused_vec_;
    delete[] thread_vec_;
    delete max_cv_;
    delete max_lock_;
    delete get_lock_;
    delete repeat_get_lock_;
  }

  void start() {
    ASSERT(queue_.isEmpty());
    finished_.store(false);
    cpu_set_t cpuset;
    for (int i = 0; i < num_threads_; ++i) {
      paused_vec_[i] = false;
      thread_vec_[i] = new thread(&Parallelizer::work, this, i);
      if (bind_method_ != 0) {
        CPU_ZERO(&cpuset);
        CPU_SET(
            (64 * ((i + num_threads_base_) % 2)) + (i + num_threads_base_) / 2,
            &cpuset);
        ASSERT(0 == pthread_setaffinity_np(thread_vec_[i]->native_handle(),
                                           sizeof(cpu_set_t), &cpuset));
      }
    }
  }

  shared_ptr<T> get() override {
    // SPDLOG_WARN("Calling get at Parallelizer: {}", fmt::ptr(this));
    get_lock_->lock();
    if (finished_.load()) {
      get_lock_->unlock();
      return nullptr;
    }
    in_flight_--;
    max_cv_->notify_one();
    shared_ptr<T> item = std::get<1>(queue_.blocking_pop());
    while (item == nullptr) {
      ++num_finished_;
      if (num_finished_ == num_threads_) {
        num_finished_ = 0;
        get_lock_->unlock();
        finished_.store(true);
        return nullptr;
      }
      item = std::get<1>(queue_.blocking_pop());
    }
    get_lock_->unlock();
    // SPDLOG_WARN("Got item at Parallelizer: {}", fmt::ptr(this));
    return item;
  }

  void stop() {
    for (int i = 0; i < num_threads_; ++i) paused_vec_[i] = true;
    num_finished_ = 0;
    finished_.store(true);
    max_cv_->notify_all();
    in_flight_ = 0;
    for (int i = 0; i < num_threads_; ++i) {
      if (thread_vec_[i]) {
        thread_vec_[i]->join();
        delete thread_vec_[i];
        thread_vec_[i] = nullptr;
      }
    }
    queue_.flush();
    in_flight_ = 0;
  }

  void work(int id) {
    Worker<T> *worker = worker_vec_[id];
    atomic<bool> &paused = paused_vec_[id];
    while (!paused) {
      std::unique_lock<std::mutex> lock(*max_lock_);
      if (in_flight_.load(std::memory_order_acquire) >= max_in_flight_) {
        max_cv_->wait(lock);
        lock.unlock();
        continue;
      } else {
        lock.unlock();
      }
      shared_ptr<T> item = worker->get();
      queue_.blocking_push(item);
      if (item == nullptr)
        paused = true;
      else
        in_flight_++;
    }
  }

  int getMaxInFlight() const { return max_in_flight_; }

  void setRepeat(int out_repeat) {
    total_repeat = out_repeat;
    ASSERT(total_repeat > 1);
  }

  shared_ptr<T> repeatedGet() {
    repeat_get_lock_->lock();
    if (curr_repeat >= total_repeat) curr_repeat = 0;
    if (curr_repeat == 0) {
      repeat_ret = this->get();
    }
    ++curr_repeat;
    auto tmp_ret = repeat_ret;
    repeat_get_lock_->unlock();
    return tmp_ret;
  }

 private:
  Queue<shared_ptr<T>> queue_;
  int num_threads_ = 0, num_finished_ = 0;
  int num_threads_base_ = 0;
  int bind_method_ = 0;
  int max_in_flight_ = 0;
  atomic<int> in_flight_;
  std::mutex *max_lock_;
  std::mutex *get_lock_;
  std::mutex *repeat_get_lock_;
  std::condition_variable *max_cv_ = nullptr;
  atomic<bool> *paused_vec_;
  atomic<bool> finished_;
  thread **thread_vec_;
  std::vector<Worker<T> *> worker_vec_;
  int total_repeat = 1;
  atomic<int> curr_repeat;
  shared_ptr<T> repeat_ret = nullptr;
};
#endif  // __PARALLEL_H__