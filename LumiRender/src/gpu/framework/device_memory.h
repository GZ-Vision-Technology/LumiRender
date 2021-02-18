//
// Created by Zero on 2021/2/12.
//


#pragma once

#include "helper/optix.h"
#include <vector>
#include <cuda_runtime.h>
#include "graphics/lstd/lstd.h"
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace luminous {
    inline namespace gpu {

        class CUDAMemoryResource : public lstd::pmr::memory_resource {
            void *do_allocate(size_t size, size_t alignment);
            void do_deallocate(void *p, size_t bytes, size_t alignment);

            bool do_is_equal(const memory_resource &other) const noexcept {
                return this == &other;
            }
        };

        class CUDATrackedMemoryResource : public CUDAMemoryResource {
        public:
            void *do_allocate(size_t size, size_t alignment);
            void do_deallocate(void *p, size_t bytes, size_t alignment);

            bool do_is_equal(const memory_resource &other) const noexcept {
                return this == &other;
            }

            void PrefetchToGPU() const;
            size_t BytesAllocated() const { return bytesAllocated; }

        private:
            bool bypassSlab(size_t size) const {
#ifdef DEBUG_BUILD
                return true;
#else
                return size > slabSize / 4;
#endif
            }

            void *cudaAllocate(size_t size, size_t alignment);

            size_t bytesAllocated = 0;
            uint8_t *currentSlab = nullptr;
            static constexpr int slabSize = 1024 * 1024;
            size_t slabOffset = slabSize;
            mutable std::mutex mutex;
            std::unordered_map<void *, size_t> allocations;
        };
    }
}