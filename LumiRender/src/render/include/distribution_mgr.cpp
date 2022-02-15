//
// Created by Zero on 2021/7/28.
//


#include "distribution_mgr.h"

namespace luminous {
    inline namespace render {

        void DistributionMgr::add_distribution2d(const vector<float> &f, int u, int v) {

            Distribution2DBuilder builder = Distribution2D::create_builder(f.data(), u, v);
            for (const auto &builder_1D : builder.conditional_v) {
                add_distribution(builder_1D);
            }
            add_distribution(builder.marginal);
        }

        void DistributionMgr::add_distribution(const Distribution1D::Builder &builder, bool need_count) {
#if USE_ALIAS_TABLE
            _handles.emplace_back(_func_buffer.size(), builder.func.size(),
                                  builder.func_integral);
            _alias_entry_buffer.append(builder.table);
#else
            _handles.emplace_back(_func_buffer.size(), builder.func.size(),
                                  _CDF_buffer.size(), builder.CDF.size(),
                                  builder.func_integral);
            _CDF_buffer.append(builder.CDF);
#endif
            _func_buffer.append(builder.func);
            if (need_count) {
                ++_count_distribution;
            }
        }

        void DistributionMgr::init_on_device(Device *device) {
            if (_handles.empty()) {
                return;
            }
            distribution2ds.clear();
            distributions.clear();
            _func_buffer.allocate_device();
            _func_buffer.synchronize_to_device();

#if USE_ALIAS_TABLE
            _alias_entry_buffer.allocate_device();
            _alias_entry_buffer.synchronize_to_device();
            for (const auto &handle : _handles) {
                BufferView<const float> func = _func_buffer.device_buffer_view(handle.offset, handle.size);
                BufferView<const AliasEntry> alias_entries = _alias_entry_buffer.device_buffer_view(handle.offset,
                                                                                                    handle.size);
                distributions.emplace_back(alias_entries, func, handle.integral);
            }
#else
            _CDF_buffer.allocate_device();
            _CDF_buffer.synchronize_to_device();
            for (const auto &handle : _handles) {
                BufferView<const float> func = _func_buffer.device_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = _CDF_buffer.device_buffer_view(handle.CDF_offset, handle.CDF_size);
                distributions.emplace_back(func, CDF, handle.integral);
            }
#endif
            distributions.allocate_device();
            int count = distributions.size() - _count_distribution;
            if (count > 0) {
                auto distribution = Distribution2D(distributions.const_device_buffer_view(_count_distribution, count),
                                                   distributions.back());
                distribution2ds.push_back(distribution);
                distribution2ds.allocate_device();
            }
        }

        void DistributionMgr::init_on_host() {
            if (_handles.empty()) {
                return;
            }
            distribution2ds.clear();
            distributions.clear();
#if USE_ALIAS_TABLE
            for (const auto &handle : _handles) {
                BufferView<const float> func = _func_buffer.const_host_buffer_view(handle.offset, handle.size);
                BufferView<const AliasEntry> alias_entries = _alias_entry_buffer.const_host_buffer_view(handle.offset,
                                                                                                    handle.size);
                distributions.emplace_back(alias_entries, func, handle.integral);
            }
#else
            for (const auto &handle : _handles) {
                BufferView<const float> func = _func_buffer.const_host_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = _CDF_buffer.const_host_buffer_view(handle.CDF_offset, handle.CDF_size);
                distributions.emplace_back(func, CDF, handle.integral);
            }
#endif
            int count = distributions.size() - _count_distribution;
            if (count > 0) {
                auto distribution = Distribution2D(distributions.const_host_buffer_view(_count_distribution, count),
                                                   distributions.back());
                distribution2ds.push_back(distribution);
            }
        }

        void DistributionMgr::shrink_to_fit() {
            _func_buffer.shrink_to_fit();
            _handles.shrink_to_fit();
#if USE_ALIAS_TABLE
            _alias_entry_buffer.shrink_to_fit();
#else
            _CDF_buffer.shrink_to_fit();
#endif
            distribution2ds.shrink_to_fit();
            distributions.shrink_to_fit();
        }

        void DistributionMgr::clear() {
#if USE_ALIAS_TABLE
            _alias_entry_buffer.clear();
#else
            _CDF_buffer.clear();
#endif
            _func_buffer.clear();

            _handles.clear();
            distribution2ds.clear();
            distributions.clear();

        }

        size_t DistributionMgr::size_in_bytes() const {
            size_t ret = 0u;
            ret += distribution2ds.size_in_bytes();
            ret += distributions.size_in_bytes();
            ret += _handles.size() * sizeof(DistributionHandle);

#if USE_ALIAS_TABLE
            ret += _alias_entry_buffer.size_in_bytes();
#else
            ret += _CDF_buffer.size_in_bytes();
#endif
            ret += _func_buffer.size_in_bytes();

            return ret;
        }

        void DistributionMgr::synchronize_to_device() {
            if (_handles.empty()) {
                return;
            }
            distribution2ds.synchronize_to_device();
            distributions.synchronize_to_device();
        }

    }
}