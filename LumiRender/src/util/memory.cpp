//
// Created by Zero on 2020/11/6.
//

#include "memory.h"

namespace luminous {
    inline namespace utility {
        /**
         * 对齐分配内存
         */
        void *alloc_aligned(size_t size) {
        #if defined(HAVE_ALIGNED_MALLOC)
            return _aligned_malloc(size, L1_CACHE_LINE_SIZE);
        #elif defined(HAVE_POSIX_MEMALIGN)
            void * ptr;
            if (posix_memalign(&ptr, L1_CACHE_LINE_SIZE, size) != 0) {
                ptr = nullptr;
            }
            return ptr;
        #else
            return memalign(L1_CACHE_LINE_SIZE, size);
        #endif
        }

        void free_aligned(void *ptr) {
            if (!ptr) {
                return;
            }
        #if defined(HAVE_ALIGNED_MALLOC)
            _aligned_free(ptr);
        #else
            free(ptr);
        #endif
        }
    }
}