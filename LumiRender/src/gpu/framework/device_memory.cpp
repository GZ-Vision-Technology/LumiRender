//
// Created by Zero on 2021/2/15.
//

#include "device_memory.h"

namespace luminous {
    inline namespace gpu {
        void *CUDAMemoryResource::do_allocate(size_t size, size_t alignment) {
            void *ptr;
            CUDA_CHECK(cudaMallocManaged(&ptr, size));
            DCHECK_EQ(0, intptr_t(ptr) % alignment);
            return ptr;
        }

        void CUDAMemoryResource::do_deallocate(void *p, size_t bytes, size_t alignment) {
            CUDA_CHECK(cudaFree(p));
        }


        void *CUDATrackedMemoryResource::do_allocate(size_t size, size_t alignment) {
            if (size == 0)
                return nullptr;

            std::lock_guard<std::mutex> lock(mutex);

            // GPU cache line alignment to avoid false sharing...
            alignment = std::max<size_t>(128, alignment);

            if (bypassSlab(size))
                return cudaAllocate(size, alignment);

            if ((slabOffset % alignment) != 0)
                slabOffset += alignment - (slabOffset % alignment);

            if (slabOffset + size > slabSize) {
                currentSlab = (uint8_t *) cudaAllocate(slabSize, 128);
                slabOffset = 0;
            }

            uint8_t *ptr = currentSlab + slabOffset;
            slabOffset += size;
            return ptr;
        }

        void *CUDATrackedMemoryResource::cudaAllocate(size_t size, size_t alignment) {
            void *ptr;
            CUDA_CHECK(cudaMallocManaged(&ptr, size));
            DCHECK_EQ(0, intptr_t(ptr) % alignment);

            allocations[ptr] = size;
            bytesAllocated += size;
            return ptr;
        }

        void CUDATrackedMemoryResource::do_deallocate(void *p, size_t size, size_t alignment) {
            if (!p)
                return;

            if (bypassSlab(size)) {
                CUDA_CHECK(cudaFree(p));

                std::lock_guard<std::mutex> lock(mutex);
                auto iter = allocations.find(p);
                DCHECK(iter != allocations.end());
                allocations.erase(iter);
                bytesAllocated -= size;
            }
            // Note: no deallocation is done if it is in a slab...
        }

        void CUDATrackedMemoryResource::PrefetchToGPU() const {
            int deviceIndex;
            CUDA_CHECK(cudaGetDevice(&deviceIndex));

            std::lock_guard<std::mutex> lock(mutex);

            size_t bytes = 0;
            for (auto iter : allocations) {
                CUDA_CHECK(
                        cudaMemPrefetchAsync(iter.first, iter.second, deviceIndex, 0 /* stream */));
                bytes += iter.second;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        static CUDATrackedMemoryResource cudaTrackedMemoryResource;
        lstd::Allocator gpuMemoryAllocator(&cudaTrackedMemoryResource);

    }
}