//
// Created by Zero on 30/08/2021.
//


#pragma once

#include "core/concepts.h"
#include "base_libs/math/common.h"

namespace luminous {
    inline namespace render{
        class Accelerator : public Noncopyable {
        public:
            NDSC virtual uint64_t handle() const = 0;


        };
    }
}