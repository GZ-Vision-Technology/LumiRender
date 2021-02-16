//
// Created by Zero on 2021/2/16.
//


#pragma once

#include "core/context.h"

namespace luminous {
    inline namespace gpu {
        class CUDADevice {
        private:
            Context * _context(nullptr);
        public:
            CUDADevice(Context *c): _context(c) {}


        };
    }
}