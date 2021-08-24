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
            DevicePtrTable _device_ptr_table;

            ProgramGroupTable _program_group_table{};

            OptixShaderBindingTable _sbt{};


        public:
            MegakernelOptixAccel(const SP<Device> &device, const GPUScene *gpu_scene, Context *context);

            ProgramGroupTable create_program_groups(OptixModule optix_module,OptixDeviceContext optix_device_context);

            OptixPipeline create_pipeline();

            void create_sbt(ProgramGroupTable program_group_table, const GPUScene *gpu_scene);

            ~MegakernelOptixAccel();

            void launch(uint2 res, Managed<LaunchParams> &launch_params);

            void clear() override;
        };
    }
}