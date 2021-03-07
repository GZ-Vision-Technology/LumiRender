//
// Created by Zero on 2021/1/10.
//

#include "optix_accel.h"

namespace luminous {
    inline namespace gpu {

        void OptixAccel::create_module() {

        }

        OptixTraversableHandle OptixAccel::build_bvh(const std::vector<OptixBuildInput> &build_inputs) {
            return 0;
        }

        void OptixAccel::build(const SP<SceneGraph> &graph) {

        }

        OptixAccel::OptixAccel(const SP<Device> &device)
        : _device(device),
        _dispatcher(_device->new_dispatcher()) {

        }


    }
}