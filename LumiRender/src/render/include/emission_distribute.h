//
// Created by Zero on 2021/4/11.
//


#pragma once

#include "graphics/math/common.h"

namespace luminous {
    inline namespace render {
        struct EmissionDistributeData {
            struct DistributeHandle {
                DistributeHandle() = default;

                DistributeHandle(uint func_offset,
                                 uint func_count,
                                 uint CDF_offset,
                                 uint CDF_count,
                                 uint integral)
                        : func_offset(func_offset),
                          func_count(func_count),
                          CDF_offset(CDF_offset),
                          CDF_count(CDF_count),
                          integral(integral) {}

                uint func_offset;
                uint func_count;
                uint CDF_offset;
                uint CDF_count;
                uint integral;
            };

            Managed<float> func;
            Managed<float> CDF;
            vector<DistributeHandle> handles;
            Managed<Distribute1D> emission_distributes;
        };
    }
}