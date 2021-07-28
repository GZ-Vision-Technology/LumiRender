//
// Created by Zero on 2021/4/11.
//


#pragma once

#include "distribution_handle.h"
#include "core/backend/managed.h"


namespace luminous {
    inline namespace render {
        struct EmissionDistribution {

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