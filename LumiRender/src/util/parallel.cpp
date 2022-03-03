//
// Created by Zero on 2020/11/3.
//

#include "parallel.h"

#include <memory>
#include <mutex>
#include <thread>

namespace luminous {
    inline namespace utility {

        void ParallelForWorkPool::init() {
            auto n = num_work_threads();
            for (uint32_t tid = 0; tid < n; tid++) {
                threads.emplace_back([=]() {
                    while (!stopped) {
                        std::unique_lock<std::mutex> lock(work_mutex);
                        while (works.empty() && !stopped) {
                            has_work.wait(lock);
                        }
                        if (stopped)
                            return;
                        auto &loop = works.front();
                        auto id = work_id;
                        lock.unlock();
                        // lock held
                        while (!loop.done()) {

                            auto begin = loop.work_index.fetch_add(loop.chunk_size);
                            for (auto i = begin; i < begin + loop.chunk_size && i < loop.count; i++) {
                                (loop.func)(i, tid);
                            }

                        }
                        lock.lock();
                        n_thread_finished++;
                        one_thread_finished.notify_all();

                        while (n_thread_finished != (uint32_t) threads.size() && work_id == id) {
                            one_thread_finished.wait(lock);
                        }

                        if (work_id == id) {
                            work_id++; // only one thread would reach here
                            works.pop_front();
                            if (works.empty()) {
                                main_waiting.notify_one();
                            }
                            n_thread_finished = 0;
                        }
                        lock.unlock();
                    }
                });
            }
        }

        void ParallelForWorkPool::enqueue(const ParallelForContext &context) {
            std::lock_guard<std::mutex> lock(work_mutex);
            works.emplace_back(context);
            has_work.notify_all();
        }

        void ParallelForWorkPool::wait() {
            std::unique_lock<std::mutex> lock(work_mutex);
            while (!works.empty()) {

                main_waiting.wait(lock);
            }
        }

        ParallelForWorkPool::~ParallelForWorkPool() {
            stopped = true;
            has_work.notify_all();
            for (auto &thr : threads) {
                thr.join();
            }
        }

        static int n_thread{0};
        void set_thread_num(int num) { n_thread = num; }

        int num_work_threads() {
            return n_thread == 0 ? std::thread::hardware_concurrency() : n_thread;
        }

        namespace thread_internal {
            static std::once_flag flag;
            static std::unique_ptr<ParallelForWorkPool> pool;
        } // namespace thread_internal

        ParallelForWorkPool *work_pool() { return thread_internal::pool.get(); }

        void init_thread_pool() {
            using namespace thread_internal;
            std::call_once(flag, [&]() { pool = std::make_unique<ParallelForWorkPool>(); });
        }

        void series(int count, const std::function<void(uint32_t, uint32_t)> &func) {
            for (int i = 0; i < count; ++i) {
                func(i, 0);
            }
        }

        void async(int count, const std::function<void(uint32_t, uint32_t)> &func, size_t chunk_size) {
            using namespace thread_internal;
            init_thread_pool();
            if (n_thread == 1) {
                series(count, func);
            }
            ParallelForContext ctx;
            ctx.func = func;
            ctx.chunk_size = (uint32_t) chunk_size;
            ctx.count = count;
            ctx.work_index = 0;
            pool->enqueue(ctx);
        }

        void parallel_for(int count, const std::function<void(uint32_t, uint32_t)> &func, size_t chunk_size) {
            using namespace thread_internal;
            async(count, func, chunk_size);
            pool->wait();
        }

        namespace thread {
            void finalize() {
                using namespace thread_internal;
                pool.reset(nullptr);
            }
        } // namespace thread

    }
}