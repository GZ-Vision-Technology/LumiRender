//
// Created by Zero on 2021/4/12.
//

#include "emission_distribution.h"

namespace luminous {
    inline namespace render {


        void EmissionDistribution::add_distribute(const Distribution1DBuilder &builder) {

            handles.emplace_back(func_buffer.size(),builder.func.size(),
                                 CDF_buffer.size(), builder.CDF.size(),
                                 builder.func_integral);
            func_buffer.append(builder.func);
            CDF_buffer.append(builder.CDF);
        }

        void EmissionDistribution::init_on_host() {
            for (const auto &handle : handles) {
                BufferView<const float> func = func_buffer.host_buffer_view(handle.func_offset, handle.func_size);
                BufferView<const float> CDF = CDF_buffer.host_buffer_view(handle.CDF_offset, handle.CDF_size);
                emission_distributions.emplace_back(func, CDF, handle.integral);
            }
        }

        void EmissionDistribution::init_on_device(const SP<Device> &device) {
            func_buffer.allocate_device(device);
            func_buffer.synchronize_to_gpu();
            CDF_buffer.allocate_device(device);
            CDF_buffer.synchronize_to_gpu();
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
    } // luminous::render
} // luminous