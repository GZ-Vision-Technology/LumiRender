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

        struct AliasTableHandle {
        public:
            size_t offset;
            size_t size;
        };

        class DistributionMgr {
        protected:
            Device * _device;
            Managed<float> _func_buffer{_device};
            Managed<float> _CDF_buffer{_device};
            /**
             * count of distribution1d
             */
            size_t _count_distribution{0};
            /**
             * the first _count_distribution is independent distribution1d
             * and the rest are distribution1d make up the distribution2d
             */
            std::vector<DistributionHandle> _handles;

            Managed<AliasEntry> _alias_entry_buffer{_device};
            Managed<float> _alias_PMF_buffer{_device};

            std::vector<AliasTableHandle> _alias_table_handles;

        public:
#if USE_ALIAS_TABLE
            Managed<AliasTable> alias_tables{_device};
            Managed<AliasTable2D> alias_table2ds{_device};
#else
            Managed<Distribution1D> distributions{_device};
            Managed<Dichotomy2D> distribution2ds{_device};
#endif
            explicit DistributionMgr(Device *device) : _device(device) {}

            void add_distribution(const DichotomyBuilder &builder, bool need_count = false);

            void add_distribution2d(const vector<float> &f, int nu, int nv);

            void add_distribution(const AliasTableBuilder &builder, bool need_count = false);

            void init_on_host();

            void synchronize_to_device();

            void init_on_device(Device *device);

            void shrink_to_fit();

            void clear();

            LM_NODISCARD virtual size_t size_in_bytes() const;
        };
    }
}