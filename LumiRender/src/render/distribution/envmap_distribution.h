//
// Created by Zero on 2021/7/27.
//


#pragma once

#include "distribution_mgr.h"

namespace luminous {
    inline namespace render {
        using std::vector;
        struct EnvmapDistribution : public DistributionMgr {
            Managed<Distribution1D> distributions;
            Managed<Distribution2D> distribution2d;

            EnvmapDistribution() = default;

            void add_distribution2d(const vector<float> &f, int nu, int nv);

            NDSC_INLINE Distribution2D get_distribution() const {
                return distribution2d.front();
            }

            void init_on_host();

            void init_on_device(const SP<Device> &device) override;

            void synchronize_to_gpu();

            void shrink_to_fit() override;

            void clear() override;

            NDSC size_t size_in_bytes() const override;
        };
    }
}