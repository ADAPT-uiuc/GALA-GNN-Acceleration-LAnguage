#ifndef QUEUE_H
#define QUEUE_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <tuple>

#include "spdlog/spdlog.h"

using std::atomic;
using std::condition_variable;
using std::deque;
using std::forward_as_tuple;
using std::mutex;
using std::tuple;
using std::unique_lock;
using namespace std::chrono_literals;

// using std::get;

template <class T>
class Queue {
 private:
  uint max_size_;

 public:
  deque<T> queue_;
  mutex *mutex_;
  condition_variable *cv_;

  Queue<T>(uint max_size = 5000);

  bool push(T item) {
    bool result = true;
    if (isFull()) {
      result = false;
    } else {
      queue_.push_back(item);
    }
    return result;
  }

  void blocking_push(T item) {
    bool pushed = false;
    while (!pushed) {
      unique_lock<mutex> lock(*mutex_);
      pushed = push(item);
      if (!pushed) {
        cv_->wait(lock);
      } else {
        cv_->notify_all();
      }
      lock.unlock();
    }
  }

  tuple<bool, T> pop() {
    bool result = true;
    T item;
    if (isEmpty()) {
      result = false;
    } else {
      item = queue_.front();
      queue_.pop_front();
    }
    return forward_as_tuple(result, item);
  }

  tuple<bool, T> blocking_pop() {
    bool popped = false;
    T item;
    while (!popped) {
      unique_lock<mutex> lock(*mutex_);
      auto tup = pop();
      popped = std::get<0>(tup);
      item = std::get<1>(tup);
      if (!popped) {
        // To debug DeadLock
        // if (cv_->wait_for(lock, 2000ms) == std::cv_status::timeout) {
        //   throw std::runtime_error("Dead lock");
        // }
        cv_->wait(lock);
      } else {
        cv_->notify_all();
      }
      lock.unlock();
    }
    return forward_as_tuple(popped, item);
  }

  void lock() { mutex_->lock(); }

  void unlock() { mutex_->unlock(); }

  void flush() {
    lock();
    queue_ = deque<T>();
    unlock();
  }

  int size() { return queue_.size(); }

  bool isFull() { return queue_.size() == max_size_; }

  bool isEmpty() { return queue_.size() == 0; }

  uint getMaxSize() { return max_size_; }

  typedef typename std::deque<T> queue_type;

  typedef typename queue_type::iterator iterator;
  typedef typename queue_type::const_iterator const_iterator;

  inline iterator begin() noexcept { return queue_.begin(); }

  inline const_iterator cbegin() const noexcept { return queue_.cbegin(); }

  inline iterator end() noexcept { return queue_.end(); }

  inline const_iterator cend() const noexcept { return queue_.cend(); }
};

template <class T>
Queue<T>::Queue(uint max_size) {
  queue_ = deque<T>();
  max_size_ = max_size;
  mutex_ = new mutex();
  cv_ = new condition_variable();
}

#endif