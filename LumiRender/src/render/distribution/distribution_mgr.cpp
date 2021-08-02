//
// Created by Zero on 2021/7/28.
//


#include "distribution_mgr.h"

namespace luminous {
    inline namespace render {

        void DistributionMgr::add_distribution(const Distribution1DBuilder &builder) {
            handles.emplace_back(func_buffer.size(), builder.func.size(),
                                 CDF_buffer.size(), builder.CDF.size(),
                                 builder.func_integral);
            func_buffer.append(builder.func);
            CDF_buffer.append(builder.CDF);
        }

        void DistributionMgr::init_on_device(const SP<Device> &device) {
            func_buffer.allocate_device(device);
            func_buffer.synchronize_to_gpu();
            CDF_buffer.allocate_device(device);
            CDF_buffer.synchronize_to_gpu();
        }

        void DistributionMgr::shrink_to_fit() {
            func_buffer.shrink_to_fit();
            CDF_buffer.shrink_to_fit();
            handles.shrink_to_fit();
        }

        void DistributionMgr::clear() {
            func_buffer.clear();
            CDF_buffer.clear();
            handles.clear();
        }

        size_t DistributionMgr::size_in_bytes() const {
            size_t ret = func_buffer.size_in_bytes();
            ret += CDF_buffer.size_in_bytes();
            ret += handles.size() * sizeof(DistributionHandle);
            return ret;
        }

    }
}