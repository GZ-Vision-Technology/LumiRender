//
// Created by Zero on 2021/7/28.
//


#pragma once

#include "base_libs/sampling/distribution.h"
#include "core/backend/managed.h"

namespace luminous {
    inline namespace render {

#if USE_ALIAS_TABLE

        struct DistributionHandle {
        public:
            size_t offset;
            size_t size;
            float integral;

            DistributionHandle(size_t offset,
                               size_t size,
                               float integral)
                    : offset(offset),
                      size(size),
                      integral(integral) {}
        };

#else
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
#endif

        class DistributionMgr {
        protected:
            Device *_device;
#if  USE_ALIAS_TABLE
            Managed<AliasEntry> _alias_entry_buffer{_device};
#else
            Managed<float> _CDF_buffer{_device};
#endif
            Managed<float> _func_buffer{_device};

            /**
             * the first _count_distribution is independent distribution1d
             * and the rest are distribution1d make up the distribution2d
             */
            std::vector<DistributionHandle> _handles;

            /**
             * count of distribution1d
             */
            size_t _count_distribution{0};

        public:

            Managed<Distribution1D> distributions{_device};
            Managed<Distribution2D> distribution2ds{_device};

            explicit DistributionMgr(Device *device) : _device(device) {}

            void add_distribution(const Distribution1D::Builder &builder, bool need_count = false);

            void add_distribution2d(const vector<float> &f, int nu, int nv);

            void init_on_host();

            void synchronize_to_device();

            void init_on_device(Device *device);

            void shrink_to_fit();

            void clear();

            LM_NODISCARD virtual size_t size_in_bytes() const;
        };
    }
}