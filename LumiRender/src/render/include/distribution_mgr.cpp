//
// Created by Zero on 2021/7/28.
//


#include "distribution_mgr.h"

namespace luminous {
    inline namespace render {

        void DistributionMgr::add_distribution2d(const vector<float> &f, int u, int v) {
#if USE_ALIAS_TABLE
            AliasTable2DBuilder builder2d = AliasTable2D::create_builder(f.data(), u, v);
            for (const auto &builder_1d : builder.conditional_v) {
                add_distribution(builder_1d);
            }
            add_distribution(builder2d.marginal);
#else
            Distribution2DBuilder builder = Distribution2D::create_builder(f.data(), u, v);
            for (const auto& builder_1D : builder.conditional_v) {
                add_distribution(builder_1D);
            }
            add_distribution(builder.marginal);
#endif
        }

        void DistributionMgr::add_distribution(const Distribution1DBuilder &builder, bool need_count) {
            _handles.emplace_back(_func_buffer.size(), builder.func.size(),
                                 _CDF_buffer.size(), builder.CDF.size(),
                                 builder.func_integral);
            _func_buffer.append(builder.func);
            _CDF_buffer.append(builder.CDF);
            if (need_count) {
                ++_count_distribution;
            }
        }

        void DistributionMgr::add_distribution(const AliasTableBuilder &builder, bool need_count) {
            _alias_table_handles.push_back({_alias_entry_buffer.size(), builder.table.size()});
            _alias_entry_buffer.append(builder.table);
            _alias_PMF_buffer.append(builder.PMF);
            if (need_count) {
                ++_count_distribution;
            }
        }

        void DistributionMgr::init_on_device(Device *device) {
#if USE_ALIAS_TABLE
            if (_alias_table_handles.empty()) {
                return;
            }

            alias_tables.clear();
            alias_table2ds.clear();
            _alias_entry_buffer.allocate_device();
            _alias_PMF_buffer.allocate_device();
            _alias_entry_buffer.synchronize_to_device();
            _alias_PMF_buffer.synchronize_to_device();

            for (const auto &handle : _alias_table_handles) {
                BufferView<const AliasEntry> alias_entry = _alias_entry_buffer.device_buffer_view(handle.offset, handle.size);
                BufferView<const float> alias_PMF = _alias_PMF_buffer.device_buffer_view(handle.offset, handle.size);
                alias_tables.emplace_back(alias_entry, alias_PMF);
            }
            alias_tables.allocate_device();

            int count = alias_tables.size() - _count_distribution;
            if (count > 0) {
                auto distribution = AliasTable2D(alias_tables.const_device_buffer_view(_count_distribution, count),
                                                 alias_tables.back());
                alias_table2ds.push_back(distribution);
                alias_table2ds.allocate_device();
            }
#else
            if (_handles.empty()) {
                return;
            }
            distribution2ds.clear();
            distributions.clear();
            _func_buffer.allocate_device();
            _func_buffer.synchronize_to_device();
            _CDF_buffer.allocate_device();
            _CDF_buffer.synchronize_to_device();

            for (const auto &handle : _handles) {
                BufferView<const float> func = _func_buffer.device_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = _CDF_buffer.device_buffer_view(handle.CDF_offset, handle.CDF_size);
                distributions.emplace_back(func, CDF, handle.integral);
            }
            distributions.allocate_device();
            int count = distributions.size() - _count_distribution;
            if (count > 0) {
                auto distribution = Distribution2D(distributions.const_device_buffer_view(_count_distribution, count),
                                                   distributions.back());
                distribution2ds.push_back(distribution);
                distribution2ds.allocate_device();
            }
#endif
        }

        void DistributionMgr::init_on_host() {
#if USE_ALIAS_TABLE
            if (_alias_table_handles.empty()) {
                return;
            }
            alias_tables.clear();
            alias_table2ds.clear();
            for (const auto &handle : _alias_table_handles) {
                BufferView<const AliasEntry> alias_entry = _alias_entry_buffer.device_buffer_view(handle.offset, handle.size);
                BufferView<const float> alias_PMF = _alias_PMF_buffer.device_buffer_view(handle.offset, handle.size);
                alias_tables.emplace_back(alias_entry, alias_PMF);
            }

            int count = alias_tables.size() - _count_distribution;
            if (count > 0) {
                auto distribution = AliasTable2D(alias_tables.const_device_buffer_view(_count_distribution, count),
                                                 alias_tables.back());
                alias_table2ds.push_back(distribution);
                alias_table2ds.allocate_device();
            }
#else
            if (_handles.empty()) {
                return;
            }
            distribution2ds.clear();
            distributions.clear();
            for (const auto &handle : _handles) {
                BufferView<const float> func = _func_buffer.const_host_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = _CDF_buffer.const_host_buffer_view(handle.CDF_offset, handle.CDF_size);
                distributions.emplace_back(func, CDF, handle.integral);
            }
            int count = distributions.size() - _count_distribution;
            if (count > 0) {
                auto distribution = Distribution2D(distributions.const_host_buffer_view(_count_distribution, count),
                                                   distributions.back());
                distribution2ds.push_back(distribution);
            }
#endif
        }

        void DistributionMgr::shrink_to_fit() {
            _func_buffer.shrink_to_fit();
            _CDF_buffer.shrink_to_fit();
            _handles.shrink_to_fit();
            distribution2ds.shrink_to_fit();
            distributions.shrink_to_fit();
        }

        void DistributionMgr::clear() {
            _func_buffer.clear();
            _CDF_buffer.clear();
            _handles.clear();
            distribution2ds.clear();
            distributions.clear();
        }

        size_t DistributionMgr::size_in_bytes() const {
            size_t ret = _func_buffer.size_in_bytes();
            ret += _CDF_buffer.size_in_bytes();
            ret += _handles.size() * sizeof(DistributionHandle);
            ret += distribution2ds.size_in_bytes();
            ret += distributions.size_in_bytes();
            return ret;
        }

        void DistributionMgr::synchronize_to_device() {
#if USE_ALIAS_TABLE
            if (_alias_table_handles.empty()) {
                return;
            }
            alias_tables.synchronize_to_device();
            alias_table2ds.synchronize_to_device();
#else
            if (_handles.empty()) {
                return;
            }
            distribution2ds.synchronize_to_device();
            distributions.synchronize_to_device();
#endif
        }

    }
}