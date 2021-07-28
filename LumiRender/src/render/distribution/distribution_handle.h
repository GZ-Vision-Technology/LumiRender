//
// Created by Zero on 2021/7/28.
//


#pragma once

#include "distribution.h"

namespace luminous {
    inline namespace render {
        struct DistributionHandle {
            DistributionHandle() = default;

            DistributionHandle(size_t func_offset,
                               size_t func_size,
                               size_t CDF_offset,
                               size_t CDF_size,
                               float integral)
                    : func_offset(func_offset),
                      func_size(func_size),
                      CDF_offset(CDF_offset),
                      CDF_size(CDF_size),
                      integral(integral) {}

            size_t func_offset;
            size_t func_size;
            size_t CDF_offset;
            size_t CDF_size;
            float integral;
        };
    }
}