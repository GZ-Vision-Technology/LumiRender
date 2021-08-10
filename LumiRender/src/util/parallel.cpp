//
// Created by Zero on 2020/11/3.
//

#include "parallel.h"
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace luminous {
    inline namespace utility {

        struct ParallelForContext {
            std::atomic_uint32_t work_index;
            size_t count = 0;
            uint32_t chunkSize = 0;

            ParallelForContext() : work_index(0) {}

            const std::function<void(uint32_t, uint32_t)> *func = nullptr;

            bool done() const { return work_index >= count; }

            ParallelForContext(const ParallelForContext &rhs)
                    : work_index(rhs.work_index.load()), count(rhs.count), chunkSize(rhs.chunkSize), func(rhs.func) {}
        };

        struct ParallelForWorkPool {
            std::deque<ParallelForContext> works;
            std::vector<std::thread> threads;
            std::condition_variable has_work, one_thread_finished, main_waiting;
            std::mutex work_mutex;
            std::atomic_bool stopped;
            std::uint32_t work_id;
            std::uint32_t n_thread_finished;

            ParallelForWorkPool() : work_id(0), n_thread_finished(0) {
                stopped = false;
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
                                auto begin = loop.work_index.fetch_add(loop.chunkSize);
                                for (auto i = begin; i < begin + loop.chunkSize && i < loop.count; i++) {
                                    (*loop.func)(i, tid);
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

            void enqueue(const ParallelForContext &context) {
                std::lock_guard<std::mutex> lock(work_mutex);
                works.emplace_back(context);
                has_work.notify_all();
            }

            void wait() {
                std::unique_lock<std::mutex> lock(work_mutex);
                while (!works.empty()) {

                    main_waiting.wait(lock);
                }
            }

            ~ParallelForWorkPool() {
                stopped = true;
                has_work.notify_all();
                for (auto &thr : threads) {
                    thr.join();
                }
            }
        };

        namespace thread_internal {
            static std::once_flag flag;
            static std::unique_ptr<ParallelForWorkPool> pool;
        } // namespace thread_internal

        void parallel_for(int count, const std::function<void(uint32_t, uint32_t)> &func, size_t chunkSize) {
            using namespace thread_internal;
            std::call_once(flag, [&]() { pool = std::make_unique<ParallelForWorkPool>(); });
            ParallelForContext ctx;
            ctx.func = &func;
            ctx.chunkSize = (uint32_t) chunkSize;
            ctx.count = count;
            ctx.work_index = 0;
            pool->enqueue(ctx);
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