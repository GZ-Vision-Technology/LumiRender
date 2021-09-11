//
// Created by Zero on 10/09/2021.
//


#pragma once

#include <atomic>
#include <functional>
#include <condition_variable>
#include <deque>
#include <future>
#include <base_libs/geometry/box.h>
#include "base_libs/math/common.h"
#include <iostream>

namespace luminous {
    inline namespace utility {

        class ThreadPool;

        ThreadPool *thread_pool();

        void init_thread_pool(int threads_num);

        void parallelFor(int count,
                         std::function<void(uint, uint)> func,
                         int chunk_size = 1);

        extern thread_local int thread_idx;

        class Barrier {
        private:
            std::mutex _mutex;
            std::condition_variable _cv;
            int _count;
        public:
            explicit Barrier(int count) : _count(count) { DCHECK_GT(count, 0); }

            ~Barrier() { DCHECK_EQ(_count, 0); }

            void wait();
        };

        struct ParallelWork {
            std::function<void(uint, uint)> func;
            size_t count = 0;
            uint32_t chunk_size = 0;
            std::atomic_uint32_t work_index;
            int active_thread_num{0};

            bool done() const { return work_index >= count && active_thread_num == 0; }

            bool valid() const { return bool(func); }

            ParallelWork(std::function<void(uint, uint)> f, size_t c, uint32_t chunk_size)
                    : func(std::move(f)), count(c), chunk_size(chunk_size), work_index(0) {}

            ParallelWork(const ParallelWork &rhs)
                    : work_index(rhs.work_index.load()),
                      count(rhs.count),
                      chunk_size(rhs.chunk_size),
                      func(rhs.func) {}
        };

        class ThreadPool {
        private:
            int _thread_num{};
            std::deque<ParallelWork> _works;
            std::vector<std::thread> _threads;
            std::condition_variable _cv;
            std::mutex _work_mtx;
            std::atomic_bool _stopped;
        public:
            explicit ThreadPool(int num);

            void init(int num);

            void execute_work(ParallelWork &work, uint tid,
                              std::unique_lock<std::mutex> &lock);

            void work_loop(uint tid, std::unique_lock<std::mutex> &lock);

            void start_work(const ParallelWork &work);

            ~ThreadPool();
        };
    }
}