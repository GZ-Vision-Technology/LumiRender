//
// Created by Zero on 2020/11/3.
//

#include "parallel.h"

#include <memory>
#include <mutex>
#include <thread>

namespace luminous {
    inline namespace utility {

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

        void parallel_for(int count, const std::function<void(uint32_t, uint32_t)> &func, size_t chunkSize) {
            using namespace thread_internal;
            init_thread_pool();
            ParallelForContext ctx;
            ctx.func = func;
            ctx.chunkSize = (uint32_t) chunkSize;
            ctx.count = count;
            ctx.work_index = 0;
            pool->enqueue(ctx);
            pool->wait();
        }

        void async(int count, const std::function<void(uint32_t, uint32_t)> &func, size_t chunkSize) {
            using namespace thread_internal;
            init_thread_pool();
            ParallelForContext ctx;
            ctx.func = func;
            ctx.chunkSize = (uint32_t) chunkSize;
            ctx.count = count;
            ctx.work_index = 0;
            pool->enqueue(ctx);
        }

        namespace thread {
            void finalize() {
                using namespace thread_internal;
                pool.reset(nullptr);
            }
        } // namespace thread
    }
}