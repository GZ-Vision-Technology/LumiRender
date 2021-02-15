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

    }
}