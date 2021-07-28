//
// Created by Zero on 2021/7/27.
//


#pragma once

#include "distribution_mgr.h"

namespace luminous {
    inline namespace render {
        using std::vector;
        struct EnvmapDistribution : public DistributionMgr {
            Managed<Distribution2D> distribution_2D;

            EnvmapDistribution() = default;

            void init(vector<float> f, int nu, int nv);

            void init_on_host();

            void init_on_device(const SP<Device> &device) override;

            void synchronize_to_gpu();

            void shrink_to_fit() override;

            void clear() override;

            NDSC size_t size_in_bytes() const override;
        };
    }
}