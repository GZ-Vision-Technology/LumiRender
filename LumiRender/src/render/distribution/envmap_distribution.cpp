//
// Created by Zero on 2021/7/27.
//

#include "envmap_distribution.h"

namespace luminous {
    inline namespace render {

        void EnvmapDistribution::init(vector<float> f, int u, int v) {
            clear();
            Distribution2DBuilder builder = Distribution2D::create_builder(f.data(), u, v);
            for (auto builder_1D : builder.conditional_v) {
                add_distribute(builder_1D);
            }
            add_distribute(builder.marginal);
        }

        void EnvmapDistribution::add_distribute(const Distribution1DBuilder &builder) {
            handles.emplace_back(func_buffer.size(), builder.func.size(),
                                 CDF_buffer.size(), builder.CDF.size(),
                                 builder.func_integral);
            func_buffer.append(builder.func);
            CDF_buffer.append(builder.CDF);
        }

        void EnvmapDistribution::init_on_host() {
            vector<Distribution1D> distributions;
            for (const auto &handle : handles) {
                BufferView<const float> func = func_buffer.const_host_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = CDF_buffer.const_host_buffer_view(handle.CDF_offset, handle.CDF_size);
                distributions.emplace_back(func, CDF, handle.integral);
            }
            auto distribution = Distribution2D(BufferView(distributions.data(), distributions.size() - 1),
                                               distributions.back());
            distribution_2D.push_back(distribution);
        }

        void EnvmapDistribution::init_on_device(const SP<Device> &device) {
            func_buffer.allocate_device(device);
            func_buffer.synchronize_to_gpu();
            CDF_buffer.allocate_device(device);
            CDF_buffer.synchronize_to_gpu();
            vector<Distribution1D> distributions;
            for (const auto &handle : handles) {
                BufferView<const float> func = func_buffer.device_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = CDF_buffer.device_buffer_view(handle.CDF_offset, handle.CDF_size);
                distributions.emplace_back(func, CDF, handle.integral);
            }
            auto distribution = Distribution2D(BufferView(distributions.data(), distributions.size() - 1),
                                               distributions.back());
            distribution_2D.push_back(distribution);
        }

        void EnvmapDistribution::synchronize_to_gpu() {
            distribution_2D.synchronize_to_gpu();
        }

        void EnvmapDistribution::shrink_to_fit() {
            func_buffer.shrink_to_fit();
            CDF_buffer.shrink_to_fit();
            handles.shrink_to_fit();
            distribution_2D.shrink_to_fit();
        }

        void EnvmapDistribution::clear() {
            func_buffer.clear();
            CDF_buffer.clear();
            handles.clear();
            distribution_2D.clear();
        }

        size_t EnvmapDistribution::size_in_bytes() const {
            size_t ret = func_buffer.size_in_bytes();
            ret += CDF_buffer.size_in_bytes();
            ret += handles.size() * sizeof(DistributionHandle);
            ret += distribution_2D.size_in_bytes();
            return ret;
        }
    }
}