//
// Created by Zero on 2021/7/27.
//

#include "envmap_distribution.h"

namespace luminous {
    inline namespace render {

        void EnvmapDistribution::init_on_host() {

        }

        void EnvmapDistribution::init_on_device(const SP<Device> &device) {

        }

        void EnvmapDistribution::synchronize_to_gpu() {

        }

        void EnvmapDistribution::shrink_to_fit() {
            func.shrink_to_fit();
        }

        void EnvmapDistribution::clear() {
            func.clear();
        }

        size_t EnvmapDistribution::size_in_bytes() const {
            return func.size_in_bytes();
        }

        void EnvmapDistribution::init(vector<float> f, int u, int v) {

        }
    }
}