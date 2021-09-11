//
// Created by Zero on 10/09/2021.
//

#include "thread_pool.h"
#include <memory>

namespace luminous {
    inline namespace utility {

        namespace thread_internal {
            static std::once_flag _flag;
            static std::unique_ptr<ThreadPool> _thread_pool;
        }

        thread_local int thread_idx;

        ThreadPool *thread_pool() {
            return thread_internal::_thread_pool.get();
        }

        void init_thread_pool(int threads_num) {
            using namespace thread_internal;
            thread_idx = 0;
            std::call_once(_flag, [&]() { _thread_pool = std::make_unique<ThreadPool>(threads_num); });
        }

        void Barrier::wait() {
            std::unique_lock<std::mutex> lock(_mutex);
            DCHECK_GT(_count, 0);
            if (--_count == 0) {
                // This is the last thread to reach the barrier; wake up all of the
                // other ones before exiting.
                _cv.notify_all();
            } else { // Otherwise there are still threads that haven't reached it. Give
                // up the lock and wait to be notified.
                _cv.wait(lock, [this] { return _count == 0; });
            }
        }

        void parallelFor(int count, std::function<void(uint, uint)> func, int chunk_size) {
            auto pool = thread_pool();
            if (count <= chunk_size) {
                for (int i = 0; i < count; ++i) {
                    func(i, 0);
                }
                return;
            }
            pool->start_work(ParallelWork(func, count, chunk_size));
        }

        // ThreadPool
        ThreadPool::ThreadPool(int num)
                : _thread_num(num == 0 ? std::thread::hardware_concurrency() : num),
                _stopped(false) {
            init(_thread_num);
        }

        void ThreadPool::start_work(const ParallelWork& work) {
            _works.push_back(work);
//            _cv.notify_all();
            work_loop(0);

        }

        void ThreadPool::work_loop(uint tid) {
            _work_mtx.lock();
            while (!_stopped && !_works.empty()) {
                auto &work = _works.front();
                _work_mtx.unlock();
                execute_work(work, tid);
                _work_mtx.lock();
                if (work.done()) {
                    _works.pop_front();
                }
            }
            _work_mtx.unlock();
        }

        void ThreadPool::execute_work(ParallelWork &work, uint tid) {
            while (!work.done()) {
                auto begin = work.work_index.fetch_add(work.chunk_size);
                for (uint i = begin; i < begin + work.chunk_size && work.count; ++i) {
                    work.func(i, tid);
                }
            }
        }

        void ThreadPool::init(int num) {
            std::shared_ptr<Barrier> barrier = std::make_shared<Barrier>(num);
            for (int tid = 1; tid < num; ++tid) {
                _threads.emplace_back([this](int tid, std::shared_ptr<Barrier> barrier) {
                    thread_idx = tid;
                    barrier->wait();
                    barrier.reset();
                    std::unique_lock<std::mutex> lock(_work_mtx);
                    _cv.wait(lock);
//                    work_loop(tid);

                }, tid, barrier);
            }
            barrier->wait();
        }

        ThreadPool::~ThreadPool() {
            _stopped = true;
            for (auto &t : _threads) {
                t.join();
            }
        }
    }
}