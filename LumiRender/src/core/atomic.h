//
// Created by Zero on 18/09/2021.
//


#pragma once

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
    inline namespace core {

        template<typename T>
        class Atomic {
        public:
            using value_type = T;
        private:
#ifdef __CUDACC__
    #ifdef LUMINOUS_USE_LEGACY_CUDA_ATOMICS
            T _val = 0;
    #else
            cuda::atomic<T, cuda::thread_scope_device> _val{0};
    #endif
#else
            std::atomic<T> _val{0};
#endif
        public:

            LM_ND_XPU T value() {
#ifdef __CUDACC__
    #ifdef LUMINOUS_USE_LEGACY_CUDA_ATOMICS
                return _val;
    #else
                return _val.load(cuda::std::memory_order_relaxed);
    #endif
#else
                return _val.load(std::memory_order_relaxed);
#endif
            }

            LM_XPU void reset() {
#ifdef __CUDACC__
    #ifdef LUMINOUS_USE_LEGACY_CUDA_ATOMICS
                _val = 0;
    #else
                _val.store(0, cuda::std::memory_order_relaxed);
    #endif
#else
                _val.store(0, std::memory_order_relaxed);
#endif
            }

            LM_ND_XPU T fetch_add(T delta) {
#ifdef __CUDACC__
    #ifdef LUMINOUS_USE_LEGACY_CUDA_ATOMICS
                return atomicAdd(&_val, delta);
    #else
                return _val.fetch_add(delta, cuda::std::memory_order_relaxed);
    #endif
#else
                return _val.fetch_add(delta, std::memory_order_relaxed);
#endif
            }
        };

        using AtomicInt = Atomic<int>;

    }
}