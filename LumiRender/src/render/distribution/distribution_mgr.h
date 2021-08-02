//
// Created by Zero on 2021/7/28.
//


#pragma once

#include "distribution_handle.h"
#include "core/backend/managed.h"

namespace luminous {
    inline namespace render {
        struct DistributionMgr {
        protected:
            Managed<float> func_buffer;
            Managed<float> CDF_buffer;
            std::vector<DistributionHandle> handles;
        public:
            virtual void add_distribution(const Distribution1DBuilder &builder);

            virtual void init_on_device(const SP<Device> &device);

            virtual void shrink_to_fit();

            virtual void clear();

            NDSC virtual size_t size_in_bytes() const;
        };
    }
}