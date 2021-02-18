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

        struct DeviceMemory {
            size_t size_in_bytes{0};
            CUdeviceptr d_pointer{0};

            inline ~DeviceMemory() { free(); }

            inline bool allocated() const { return !empty(); }

            inline bool empty() const { return size_in_bytes == 0; }

            inline bool not_empty() const { return !empty(); }

            inline size_t size() const { return size_in_bytes; }

            inline void alloc(size_t size);

            inline void alloc_managed(size_t size);

            inline void *get();

            inline void upload(const void *h_pointer, const char *debugMessage = nullptr);

            inline void upload_async(const void *h_pointer, cudaStream_t stream);

            inline void download(void *h_pointer);

            inline void free();

            template<typename T>
            inline void upload(const std::vector<T> &vec);
        };

        inline void DeviceMemory::alloc(size_t size) {
            if (allocated()) free();

            assert(empty());
            this->size_in_bytes = size;
            CUDA_CHECK(cudaMalloc((void **) &d_pointer, size_in_bytes));
            assert(allocated() || size == 0);
        }

        inline void DeviceMemory::alloc_managed(size_t size) {
            assert(empty());
            this->size_in_bytes = size;
            CUDA_CHECK(cudaMallocManaged((void **) &d_pointer, size_in_bytes));
            assert(allocated() || size == 0);
        }

        inline void *DeviceMemory::get() {
            return (void *) d_pointer;
        }

        inline void DeviceMemory::upload(const void *h_pointer, const char *debugMessage) {
            assert(allocated() || empty());
            CUDA_CHECK(cudaMemcpy((void *) d_pointer, h_pointer,
                                   size_in_bytes, cudaMemcpyHostToDevice));
        }

        inline void DeviceMemory::upload_async(const void *h_pointer, cudaStream_t stream) {
            assert(allocated() || empty());
            CUDA_CHECK(cudaMemcpyAsync((void *) d_pointer, h_pointer,
                                       size_in_bytes, cudaMemcpyHostToDevice,
                                       stream));
        }

        inline void DeviceMemory::download(void *h_pointer) {
            assert(allocated() || size_in_bytes == 0);
            CUDA_CHECK(cudaMemcpy(h_pointer, (void *) d_pointer,
                                  size_in_bytes, cudaMemcpyDeviceToHost));
        }

        inline void DeviceMemory::free() {
            assert(allocated() || empty());
            if (!empty()) {
                CUDA_CHECK(cudaFree((void *) d_pointer));
            }
            size_in_bytes = 0;
            d_pointer = 0;
            assert(empty());
        }

        template<typename T>
        inline void DeviceMemory::upload(const std::vector<T> &vec) {
            if (!allocated()) {
                alloc(vec.size() * sizeof(T));
            } else {
                assert(size() == vec.size() * sizeof(T));
            }
            assert(allocated() || vec.empty());
            upload(vec.data());
        }

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