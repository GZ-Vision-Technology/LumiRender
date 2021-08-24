//
// Created by Zero on 2021/1/10.
//


#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include "gpu/framework/cuda_impl.h"
#include "base_libs/geometry/common.h"
#include "render/include/scene_graph.h"
#include "optix_params.h"
#include "core/backend/managed.h"
#include "optix_accel.h"

namespace luminous {
    inline namespace gpu {

        class GPUScene;

        class MegakernelOptixAccel : public OptixAccel {

        public:
            MegakernelOptixAccel(const SP<Device> &device, const GPUScene *gpu_scene, Context *context);

            ~MegakernelOptixAccel();

            void launch(uint2 res, Managed<LaunchParams> &launch_params);

            void clear() override;
        };
    }
}