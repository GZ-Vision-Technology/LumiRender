//
// Created by Zero on 2021/7/28.
//


#include "distribution_mgr.h"

namespace luminous {
    inline namespace render {

        void DistributionMgr::add_distribution2d(const vector<float> &f, int u, int v) {
            Distribution2DBuilder builder = Distribution2D::create_builder(f.data(), u, v);
            for (const auto& builder_1D : builder.conditional_v) {
                add_distribution(builder_1D);
            }
            add_distribution(builder.marginal);
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

        void DistributionMgr::init_on_device(const SP<Device> &device) {
            if (_handles.empty()) {
                return;
            }
            distribution2ds.clear();
            distributions.clear();
            _func_buffer.allocate_device(device);
            _func_buffer.synchronize_to_gpu();
            _CDF_buffer.allocate_device(device);
            _CDF_buffer.synchronize_to_gpu();

            for (const auto &handle : _handles) {
                BufferView<const float> func = _func_buffer.device_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = _CDF_buffer.device_buffer_view(handle.CDF_offset, handle.CDF_size);
                distributions.emplace_back(func, CDF, handle.integral);
            }
            int count = distributions.size() - 1 - _count_distribution;
            auto distribution = Distribution2D(distributions.const_device_buffer_view(_count_distribution, count),
                                               distributions.back());
            distribution2ds.push_back(distribution);
            distribution2ds.allocate_device(device);
            distributions.allocate_device(device);
        }

        void DistributionMgr::init_on_host() {
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
            int count = distributions.size() - 1 - _count_distribution;
            auto distribution = Distribution2D(distributions.const_device_buffer_view(_count_distribution, count),
                                               distributions.back());
            distribution2ds.push_back(distribution);
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

        void DistributionMgr::synchronize_to_gpu() {
            if (_handles.empty()) {
                return;
            }
            distribution2ds.synchronize_to_gpu();
            distributions.synchronize_to_gpu();
        }

    }
}