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

        void DistributionMgr::add_distribution(const Distribution1DBuilder &builder) {
            handles.emplace_back(func_buffer.size(), builder.func.size(),
                                 CDF_buffer.size(), builder.CDF.size(),
                                 builder.func_integral);
            func_buffer.append(builder.func);
            CDF_buffer.append(builder.CDF);
        }

        void DistributionMgr::init_on_device(const SP<Device> &device) {
            if (handles.empty()) {
                return;
            }
            distribution2ds.clear();
            distributions.clear();
            func_buffer.allocate_device(device);
            func_buffer.synchronize_to_gpu();
            CDF_buffer.allocate_device(device);
            CDF_buffer.synchronize_to_gpu();

            for (const auto &handle : handles) {
                BufferView<const float> func = func_buffer.device_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = CDF_buffer.device_buffer_view(handle.CDF_offset, handle.CDF_size);
                distributions.emplace_back(func, CDF, handle.integral);
            }
            auto distribution = Distribution2D(BufferView(distributions.data(), distributions.size() - 1),
                                               distributions.back());
            distribution2ds.push_back(distribution);
            distribution2ds.allocate_device(device);
            distributions.allocate_device(device);
        }

        void DistributionMgr::shrink_to_fit() {
            func_buffer.shrink_to_fit();
            CDF_buffer.shrink_to_fit();
            handles.shrink_to_fit();
            distribution2ds.shrink_to_fit();
            distributions.shrink_to_fit();
        }

        void DistributionMgr::clear() {
            func_buffer.clear();
            CDF_buffer.clear();
            handles.clear();
            distribution2ds.clear();
            distributions.clear();
        }

        size_t DistributionMgr::size_in_bytes() const {
            size_t ret = func_buffer.size_in_bytes();
            ret += CDF_buffer.size_in_bytes();
            ret += handles.size() * sizeof(DistributionHandle);
            ret += distribution2ds.size_in_bytes();
            ret += distributions.size_in_bytes();
            return ret;
        }

        void DistributionMgr::init_on_host() {
            distribution2ds.clear();
            distributions.clear();
            for (const auto &handle : handles) {
                BufferView<const float> func = func_buffer.const_host_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = CDF_buffer.const_host_buffer_view(handle.CDF_offset, handle.CDF_size);
                distributions.emplace_back(func, CDF, handle.integral);
            }
            auto distribution = Distribution2D(BufferView(distributions.data(), distributions.size() - 1),
                                               distributions.back());
            distribution2ds.push_back(distribution);
        }

        void DistributionMgr::synchronize_to_gpu() {
            if (handles.empty()) {
                return;
            }
            distribution2ds.synchronize_to_gpu();
            distributions.synchronize_to_gpu();
        }

    }
}