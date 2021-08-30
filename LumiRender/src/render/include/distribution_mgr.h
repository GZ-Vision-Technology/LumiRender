//
// Created by Zero on 2021/7/28.
//


#pragma once

#include "base_libs/sampling/distribution.h"
#include "core/backend/managed.h"

namespace luminous {
    inline namespace render {
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

        class DistributionMgr {
        protected:
            Managed<float> _func_buffer;
            Managed<float> _CDF_buffer;
            /**
             * count of distribution1d
             */
            size_t _count_distribution{0};
            /**
             * the first _count_distribution is independent distribution1d
             * and the rest are distribution1d make up the distribution2d
             */
            std::vector<DistributionHandle> _handles;
        public:

            Managed<Distribution1D> distributions;
            Managed<Distribution2D> distribution2ds;

            void add_distribution(const Distribution1DBuilder &builder, bool need_count = false);

            void add_distribution2d(const vector<float> &f, int nu, int nv);

            void init_on_host();

            void synchronize_to_gpu();

            void init_on_device(const SP<Device> &device);

            void shrink_to_fit();

            void clear();

            NDSC virtual size_t size_in_bytes() const;
        };
    }
}