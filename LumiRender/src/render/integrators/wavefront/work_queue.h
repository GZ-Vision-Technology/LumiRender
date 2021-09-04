//
// Created by Zero on 21/08/2021.
//


#pragma once

#include "work_items.h"

#ifdef __CUDACC__
#ifdef LUMINOUS_IS_WINDOWS
#if (__CUDA_ARCH__ < 700)
#define LUMINOUS_USE_LEGACY_CUDA_ATOMICS
#endif
#else
#if (__CUDA_ARCH__ < 600)
#define LUMINOUS_USE_LEGACY_CUDA_ATOMICS
#endif
#endif  // LUMINOUS_USE_LEGACY_CUDA_ATOMICS

#ifndef LUMINOUS_USE_LEGACY_CUDA_ATOMICS
#include <cuda/atomic>
#endif
#else

#include <atomic>
#include <utility>

#endif


namespace luminous {
    inline namespace render {
        template<typename WorkItem>
        class WorkQueue : public SOA<WorkItem> {
        private:
#ifdef __CUDACC__
#ifdef LUMINOUS_USE_LEGACY_CUDA_ATOMICS
            int _size = 0;
#else
            cuda::atomic<int, cuda::thread_scope_device> _size{0};
#endif
#else
            std::atomic<int> _size{0};
#endif
        public:
            WorkQueue() = default;

            WorkQueue(int n, Device *device)
                    : SOA<WorkItem>(n, device) {}

            WorkQueue(const WorkQueue &other)
                    : SOA<WorkItem>(other) {
#if defined(__CUDACC__) && defined(LUMINOUS_USE_LEGACY_CUDA_ATOMICS)
                size = w.size;
#else
                _size.store(other._size.load());
#endif
            }

            WorkQueue &operator=(const WorkQueue &other) {
                SOA<WorkItem>::operator=(other);
#if defined(__CUDACC__) && defined(LUMINOUS_USE_LEGACY_CUDA_ATOMICS)
                size = w.size;
#else
                _size.store(other._size.load());
#endif
                return *this;
            }

            XPU int size() const {
#ifdef __CUDACC__
    #ifdef LUMINOUS_USE_LEGACY_CUDA_ATOMICS
                return _size;
    #else
                return _size.load(cuda::std::memory_order_relaxed);
    #endif
#else
                return _size.load(std::memory_order_relaxed);
#endif
            }

            XPU void reset() {
#ifdef __CUDACC__
    #ifdef LUMINOUS_USE_LEGACY_CUDA_ATOMICS
                _size = 0;
    #else
                _size.store(0, cuda::std::memory_order_relaxed);
    #endif
#else
                _size.store(0, std::memory_order_relaxed);
#endif
            }

            XPU int allocate_entry() {
#ifdef __CUDACC__
    #ifdef LUMINOUS_USE_LEGACY_CUDA_ATOMICS
                return atomicAdd(&_size, 1);
    #else
                return _size.fetch_add(1, cuda::std::memory_order_relaxed);
    #endif
#else
                return _size.fetch_add(1, std::memory_order_relaxed);
#endif
            }

            XPU int push(WorkItem w) {
                int index = allocate_entry();
                (*this)[index] = w;
                return index;
            }
        };
    }
}