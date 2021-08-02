//
// Created by Zero on 2021/7/27.
//

#include "envmap_distribution.h"

namespace luminous {
    inline namespace render {

        void EnvmapDistribution::add_distribution2d(const vector<float> &f, int u, int v) {
            Distribution2DBuilder builder = Distribution2D::create_builder(f.data(), u, v);
            for (const auto& builder_1D : builder.conditional_v) {
                add_distribution(builder_1D);
            }
            add_distribution(builder.marginal);
        }

        void EnvmapDistribution::init_on_host() {
            distribution2d.clear();
            distributions.clear();
            for (const auto &handle : handles) {
                BufferView<const float> func = func_buffer.const_host_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = CDF_buffer.const_host_buffer_view(handle.CDF_offset, handle.CDF_size);
                distributions.emplace_back(func, CDF, handle.integral);
            }
            auto distribution = Distribution2D(BufferView(distributions.data(), distributions.size() - 1),
                                               distributions.back());
            distribution2d.push_back(distribution);
        }

        void EnvmapDistribution::init_on_device(const SP<Device> &device) {
            if (handles.empty()) {
                return;
            }
            distribution2d.clear();
            distributions.clear();
            DistributionMgr::init_on_device(device);
            for (const auto &handle : handles) {
                BufferView<const float> func = func_buffer.device_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = CDF_buffer.device_buffer_view(handle.CDF_offset, handle.CDF_size);
                distributions.emplace_back(func, CDF, handle.integral);
            }
            auto distribution = Distribution2D(BufferView(distributions.data(), distributions.size() - 1),
                                               distributions.back());
            distribution2d.push_back(distribution);
            distribution2d.allocate_device(device);
            distributions.allocate_device(device);
        }

        void EnvmapDistribution::synchronize_to_gpu() {
            if (handles.empty()) {
                return;
            }
            distribution2d.synchronize_to_gpu();
        }

        void EnvmapDistribution::shrink_to_fit() {
            DistributionMgr::shrink_to_fit();
            distributions.shrink_to_fit();
            distribution2d.shrink_to_fit();
        }

        void EnvmapDistribution::clear() {
            DistributionMgr::clear();
            distributions.clear();
            distribution2d.clear();
        }

        size_t EnvmapDistribution::size_in_bytes() const {
            size_t ret = DistributionMgr::size_in_bytes();
            ret += distribution2d.size_in_bytes();
            ret += distributions.size_in_bytes();
            return ret;
        }
    }
}