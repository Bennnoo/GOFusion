/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 * SPDX-FileCopyrightText: 2026 Anonymous GOFusion author (GOFusion modifications)
 */

/**
 * @file ThreadsafeQueue.hpp
 * @brief Header file for the ThreadsafeQueue class.
 * @author Simon Lynen
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_THREADSAFE_THREADSAFEQUEUE_HPP_
#define INCLUDE_OKVIS_THREADSAFE_THREADSAFEQUEUE_HPP_

#include <atomic>
#include <pthread.h>
#include <queue>
#include <sys/time.h>
#include <utility>

#include <glog/logging.h>

namespace okvis {

namespace threadsafe {

template<typename QueueType, class Allocator = std::allocator<QueueType>>
class Queue {

 public:

  virtual void NotifyAll() const final {
    pthread_cond_broadcast(&condition_empty_);
    pthread_cond_broadcast(&condition_full_);
  }

  Queue() {
    shutdown_ = false;
    pthread_mutex_init(&mutex_, nullptr);
    pthread_cond_init(&condition_empty_, nullptr);
    pthread_cond_init(&condition_full_, nullptr);
  }

  virtual ~Queue() {
    shutdown_ = true;
    NotifyAll();
    pthread_mutex_destroy(&mutex_);
    pthread_cond_destroy(&condition_empty_);
    pthread_cond_destroy(&condition_full_);
  }

  virtual void Shutdown() final {
    shutdown_ = true;
    NotifyAll();
  }

  virtual void Clear() final {
    decltype(queue_) discarded;
    pthread_mutex_lock(&mutex_);
    queue_.swap(discarded);
    pthread_cond_broadcast(&condition_full_);
    pthread_mutex_unlock(&mutex_);
  }

  virtual void Resume() final {
    shutdown_ = false;
    NotifyAll();
  }

  void Push(const QueueType& value) {
    PushNonBlocking(value);
  }

  void Push(QueueType&& value) {
    PushNonBlocking(std::move(value));
  }

  void PushNonBlocking(const QueueType& value) {
    pthread_mutex_lock(&mutex_);
    queue_.push(value);
    pthread_cond_signal(&condition_empty_);  
    pthread_mutex_unlock(&mutex_);
  }

  void PushNonBlocking(QueueType&& value) {
    pthread_mutex_lock(&mutex_);
    queue_.push(std::move(value));
    pthread_cond_signal(&condition_empty_);  
    pthread_mutex_unlock(&mutex_);
  }

  virtual size_t Size() const final {
    pthread_mutex_lock(&mutex_);
    size_t size = queue_.size();
    pthread_mutex_unlock(&mutex_);
    return size;
  }

  virtual bool Empty() const final {
    pthread_mutex_lock(&mutex_);
    bool empty = queue_.empty();
    pthread_mutex_unlock(&mutex_);
    return empty;
  }

  bool PushBlockingIfFull(const QueueType& value, size_t max_queue_size) {
    while (!shutdown_) {
      pthread_mutex_lock(&mutex_);
      size_t size = queue_.size();
      if (size >= max_queue_size) {
        pthread_cond_wait(&condition_full_, &mutex_);
      }
      if (size >= max_queue_size) {
        pthread_mutex_unlock(&mutex_);
        continue;
      }
      queue_.push(value);
      pthread_cond_signal(&condition_empty_);  
      pthread_mutex_unlock(&mutex_);
      return true;
    }
    return false;
  }

  bool PushBlockingIfFull(QueueType&& value, size_t max_queue_size) {
    while (!shutdown_) {
      pthread_mutex_lock(&mutex_);
      size_t size = queue_.size();
      if (size >= max_queue_size) {
        pthread_cond_wait(&condition_full_, &mutex_);
      }
      if (size >= max_queue_size) {
        pthread_mutex_unlock(&mutex_);
        continue;
      }
      queue_.push(std::move(value));
      pthread_cond_signal(&condition_empty_);  
      pthread_mutex_unlock(&mutex_);
      return true;
    }
    return false;
  }

  bool PushNonBlockingDroppingIfFull(const QueueType& value,
                                     size_t max_queue_size) {
    pthread_mutex_lock(&mutex_);
    bool result = false;
    if (queue_.size() >= max_queue_size) {
      queue_.pop();
      result = true;
    }
    queue_.push(value);
    pthread_cond_signal(&condition_empty_);  
    pthread_mutex_unlock(&mutex_);
    return result;
  }

  bool PushNonBlockingDroppingIfFull(QueueType&& value,
                                     size_t max_queue_size) {
    pthread_mutex_lock(&mutex_);
    bool result = false;
    if (queue_.size() >= max_queue_size) {
      queue_.pop();
      result = true;
    }
    queue_.push(std::move(value));
    pthread_cond_signal(&condition_empty_);  
    pthread_mutex_unlock(&mutex_);
    return result;
  }

  bool Pop(QueueType* value) {
    return PopBlocking(value);
  }

  bool PopBlocking(QueueType* value) {
    CHECK_NOTNULL(value);
    while (!shutdown_) {
      pthread_mutex_lock(&mutex_);
      if (queue_.empty()) {
        pthread_cond_wait(&condition_empty_, &mutex_);
      }
      if (queue_.empty()) {
        pthread_mutex_unlock(&mutex_);
        continue;
      }
      QueueType _value = std::move(queue_.front());
      queue_.pop();
      pthread_cond_signal(&condition_full_);  
      pthread_mutex_unlock(&mutex_);
      *value = std::move(_value);
      return true;
    }
    return false;
  }

  bool PopNonBlocking(QueueType* value) {
    CHECK_NOTNULL(value);
    pthread_mutex_lock(&mutex_);
    if (queue_.empty()) {
      pthread_mutex_unlock(&mutex_);
      return false;
    }
    *value = std::move(queue_.front());
    queue_.pop();
    pthread_mutex_unlock(&mutex_);
    return true;
  }

  bool PopTimeout(QueueType* value, int64_t timeout_nanoseconds) {
    CHECK_NOTNULL(value);
    pthread_mutex_lock(&mutex_);
    if (queue_.empty()) {
      struct timeval tv;
      struct timespec ts;
      gettimeofday(&tv, nullptr);
      ts.tv_sec = tv.tv_sec;
      ts.tv_nsec = tv.tv_usec * 1e3 + timeout_nanoseconds;
      pthread_cond_timedwait(&condition_empty_, &mutex_, &ts);
    }
    if (queue_.empty()) {
      pthread_mutex_unlock(&mutex_);
      return false;
    }
    QueueType _value = std::move(queue_.front());
    queue_.pop();
    pthread_cond_signal(&condition_full_);  
    pthread_mutex_unlock(&mutex_);
    *value = std::move(_value);
    return true;
  }

  bool getCopyOfFront(QueueType* value) {
    CHECK_NOTNULL(value);
    pthread_mutex_lock(&mutex_);
    if (queue_.empty()) {
      pthread_mutex_unlock(&mutex_);
      return false;
    }

    *value = queue_.front();
    pthread_mutex_unlock(&mutex_);
    return true;
  }

  bool getCopyOfFrontBlocking(QueueType* value) {
    CHECK_NOTNULL(value);
    while (!shutdown_) {
      pthread_mutex_lock(&mutex_);
      if (queue_.empty()) {
        pthread_cond_wait(&condition_empty_, &mutex_);
      }
      if (queue_.empty()) {
        pthread_mutex_unlock(&mutex_);
        continue;
      }
      *value = queue_.front();
      pthread_mutex_unlock(&mutex_);
      return true;
    }
    return false;
  }

  bool getCopyOfBack(QueueType* value) {
    CHECK_NOTNULL(value);
    pthread_mutex_lock(&mutex_);
    if (queue_.empty()) {
      pthread_mutex_unlock(&mutex_);
      return false;
    }

    *value = queue_.back();
    pthread_mutex_unlock(&mutex_);
    return true;
  }

  mutable pthread_mutex_t mutex_;                

  mutable pthread_cond_t condition_empty_;

  mutable pthread_cond_t condition_full_;

  std::queue<QueueType,
      std::deque<QueueType, Allocator>> queue_;  
  std::atomic_bool shutdown_;                    

};

}  

}  

#endif  
