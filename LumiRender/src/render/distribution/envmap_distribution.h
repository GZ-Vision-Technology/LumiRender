//
// Created by Zero on 2021/7/27.
//


#pragma once

#include "distribution_handle.h"
#include "core/backend/managed.h"

namespace luminous {
    inline namespace render {
        using std::vector;
        struct EnvmapDistribution {
            Managed<float> func_buffer;
            Managed<float> CDF_buffer;
            std::vector<DistributionHandle> handles;
            Managed<Distribution2D> distribution_2D;

            EnvmapDistribution() = default;

            void init(vector<float> f, int nu, int nv);

            void init_on_host();

            void init_on_device(const SP<Device> &device);

            void synchronize_to_gpu();

            void shrink_to_fit();

            void clear();

            NDSC size_t size_in_bytes() const;
        };
    }
}