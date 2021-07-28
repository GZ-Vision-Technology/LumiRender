//
// Created by Zero on 2021/4/12.
//

#include "emission_distribution.h"

namespace luminous {
    inline namespace render {

        void EmissionDistribution::init_on_host() {
            for (const auto &handle : handles) {
                BufferView<const float> func = func_buffer.const_host_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = CDF_buffer.const_host_buffer_view(handle.CDF_offset, handle.CDF_size);
                emission_distributions.emplace_back(func, CDF, handle.integral);
            }
        }

        void EmissionDistribution::init_on_device(const SP<Device> &device) {
            DistributionMgr::init_on_device(device);
            for (const auto &handle : handles) {
                BufferView<const float> func = func_buffer.device_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = CDF_buffer.device_buffer_view(handle.CDF_offset, handle.CDF_size);
                emission_distributions.emplace_back(func, CDF, handle.integral);
            }
            emission_distributions.allocate_device(device);
        }

        void EmissionDistribution::synchronize_to_gpu() {
            emission_distributions.synchronize_to_gpu();
        }

        void EmissionDistribution::shrink_to_fit() {
            DistributionMgr::shrink_to_fit();
            emission_distributions.shrink_to_fit();
        }

        void EmissionDistribution::clear() {
            DistributionMgr::clear();
            emission_distributions.clear();
        }

        size_t EmissionDistribution::size_in_bytes() const {
            size_t ret = DistributionMgr::size_in_bytes();
            ret += emission_distributions.size_in_bytes();
            return ret;
        }
    } // luminous::render
} // luminous