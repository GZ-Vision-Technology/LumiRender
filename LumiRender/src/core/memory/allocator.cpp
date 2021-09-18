//
// Created by Zero on 18/09/2021.
//

#include "allocator.h"

namespace luminous {
    inline namespace core {
        static MemoryArena s_memory_arena;

        MemoryArena& get_arena() {
            return s_memory_arena;
        }
    }
}