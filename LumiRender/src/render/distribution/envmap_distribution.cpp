//
// Created by Zero on 2021/7/27.
//

#include "envmap_distribution.h"

namespace luminous {
    inline namespace render {

        void EnvmapDistribution::init(const vector<float> &f, int u, int v) {
            clear();
            Distribution2DBuilder builder = Distribution2D::create_builder(f.data(), u, v);
            for (const auto& builder_1D : builder.conditional_v) {
                add_distribute(builder_1D);
            }
            add_distribute(builder.marginal);
        }

        void EnvmapDistribution::init_on_host() {
            distribution_2d.clear();
            distributions.clear();
            for (const auto &handle : handles) {
                BufferView<const float> func = func_buffer.const_host_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = CDF_buffer.const_host_buffer_view(handle.CDF_offset, handle.CDF_size);
                distributions.emplace_back(func, CDF, handle.integral);
            }
            auto distribution = Distribution2D(BufferView(distributions.data(), distributions.size() - 1),
                                               distributions.back());
            distribution_2d.push_back(distribution);
        }

        void EnvmapDistribution::init_on_device(const SP<Device> &device) {
            distribution_2d.clear();
            distributions.clear();
            DistributionMgr::init_on_device(device);
            for (const auto &handle : handles) {
                BufferView<const float> func = func_buffer.device_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = CDF_buffer.device_buffer_view(handle.CDF_offset, handle.CDF_size);
                distributions.emplace_back(func, CDF, handle.integral);
            }
            auto distribution = Distribution2D(BufferView(distributions.data(), distributions.size() - 1),
                                               distributions.back());
            distribution_2d.push_back(distribution);
            distribution_2d.allocate_device(device);
            distributions.allocate_device(device);
        }

        void EnvmapDistribution::synchronize_to_gpu() {
            distribution_2d.synchronize_to_gpu();
        }

        void EnvmapDistribution::shrink_to_fit() {
            DistributionMgr::shrink_to_fit();
            distributions.shrink_to_fit();
            distribution_2d.shrink_to_fit();
        }

        void EnvmapDistribution::clear() {
            DistributionMgr::clear();
            distributions.clear();
            distribution_2d.clear();
        }

        size_t EnvmapDistribution::size_in_bytes() const {
            size_t ret = DistributionMgr::size_in_bytes();
            ret += distribution_2d.size_in_bytes();
            ret += distributions.size_in_bytes();
            return ret;
        }
    }
}