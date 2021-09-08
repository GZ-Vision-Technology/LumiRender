//
// Created by Zero on 2021/1/10.
//


#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include "gpu/framework/cuda_impl.h"
#include "base_libs/geometry/common.h"
#include "render/scene/scene_graph.h"
#include "optix_params.h"
#include "core/backend/managed.h"
#include "optix_accel.h"

namespace luminous {
    inline namespace gpu {

        class MegakernelOptixAccel : public OptixAccel {
        private:
            ShaderWrapper _shader_wrapper;
        public:
            MegakernelOptixAccel(Device *device, Context *context, const Scene *scene);

            ~MegakernelOptixAccel();

            void launch(uint2 res, Managed<LaunchParams> &launch_params);

            void clear() override;
        };
    }
}