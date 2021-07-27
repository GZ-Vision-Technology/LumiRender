//
// Created by Zero on 2021/7/27.
//


#pragma once

#include "distribution.h"
#include "core/backend/managed.h"

namespace luminous {
    inline namespace render {
        using std::vector;
        struct EnvmapDistribution {
            Managed<float> func;
            int nu, nv;

            EnvmapDistribution() = default;

            void init(vector<float> func, int nu, int nv);

            void init_on_host();

            void init_on_device(const SP<Device> &device);

            void synchronize_to_gpu();

            void shrink_to_fit();

            void clear();

            NDSC size_t size_in_bytes() const;
        };
    }
}