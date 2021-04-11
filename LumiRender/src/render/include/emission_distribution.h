//
// Created by Zero on 2021/4/11.
//


#pragma once

#include "graphics/sampling/distribution.h"

namespace luminous {
    inline namespace render {
        struct EmissionDistribution {
            struct DistributionHandle {
                DistributionHandle() = default;

                DistributionHandle(uint func_offset,
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
            std::vector<DistributionHandle> handles;
            Managed<Distribution1D> emission_distributes;

            void add_distribute(const Distribution1DBuilder &builder);

            void init_on_host();

            void init_on_device();

            void synchronize_to_gpu();
        };
    }
}