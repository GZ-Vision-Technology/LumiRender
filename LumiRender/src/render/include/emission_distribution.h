//
// Created by Zero on 2021/4/11.
//


#pragma once

#include "distribution.h"
#include "core/backend/managed.h"


namespace luminous {
    inline namespace render {
        struct EmissionDistribution {
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

            Managed<float> func_buffer;
            Managed<float> CDF_buffer;
            std::vector<DistributionHandle> handles;
            Managed<Distribution1D> emission_distributions;

            void add_distribute(const Distribution1DBuilder &builder);

            void init_on_host();

            void init_on_device(const SP<Device> &device);

            void synchronize_to_gpu();

            void shrink_to_fit();

            void clear();

            NDSC size_t size_in_bytes() const;
        };
    }
}