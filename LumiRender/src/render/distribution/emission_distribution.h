//
// Created by Zero on 2021/4/11.
//


#pragma once

#include "distribution_handle.h"
#include "core/backend/managed.h"
#include "distribution_mgr.h"

namespace luminous {
    inline namespace render {
        struct EmissionDistribution : public DistributionMgr {

            Managed<Distribution1D> emission_distributions;

            void init_on_host();

            void init_on_device(const SP<Device> &device) override;

            void synchronize_to_gpu();

            void shrink_to_fit() override;

            void clear() override;

            NDSC size_t size_in_bytes() const override;
        };
    }
}