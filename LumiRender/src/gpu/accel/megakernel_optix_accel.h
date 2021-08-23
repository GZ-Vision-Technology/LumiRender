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
        private:
            Context *_context{};
            OptixPipeline _optix_pipeline{};
            OptixModule _optix_module{};
            OptixPipelineCompileOptions _pipeline_compile_options = {};
            uint32_t geom_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

            struct ProgramGroupTable {
                OptixProgramGroup raygen_prog_group{nullptr};
                OptixProgramGroup radiance_miss_group{nullptr};
                OptixProgramGroup occlusion_miss_group{nullptr};
                OptixProgramGroup radiance_hit_group{nullptr};
                OptixProgramGroup occlusion_hit_group{nullptr};

                static constexpr auto size() {
                    return sizeof(ProgramGroupTable) / sizeof(OptixProgramGroup);
                }

                void clear() {
                    optixProgramGroupDestroy(raygen_prog_group);
                    optixProgramGroupDestroy(radiance_miss_group);
                    optixProgramGroupDestroy(occlusion_miss_group);
                    optixProgramGroupDestroy(radiance_hit_group);
                    optixProgramGroupDestroy(occlusion_hit_group);
                }
            };

            struct DevicePtrTable {
                Buffer<RayGenRecord> rg_record{nullptr};
                Buffer<SceneRecord> miss_record{nullptr};
                Buffer<SceneRecord> hit_record{nullptr};
            };

            DevicePtrTable _device_ptr_table;

            ProgramGroupTable _program_group_table{};

            OptixShaderBindingTable _sbt{};

        private:

            OptixModule create_module();

            ProgramGroupTable create_program_groups(OptixModule optix_module);

            OptixPipeline create_pipeline();

            void create_sbt(ProgramGroupTable program_group_table, const GPUScene *gpu_scene);

        public:
            MegakernelOptixAccel(const SP<Device> &device, const GPUScene *gpu_scene, Context *context);

            ~MegakernelOptixAccel();

            void launch(uint2 res, Managed<LaunchParams> &launch_params);

            void clear() override;
        };
    }
}