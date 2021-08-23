//
// Created by Zero on 23/08/2021.
//


#pragma once

#include <optix.h>
#include "core/concepts.h"
#include "core/backend/buffer.h"
#include "optix_params.h"
#include "gpu/framework/cuda_impl.h"

namespace luminous {
    inline namespace gpu {

        class GPUScene;

        struct ProgramName {
            char *raygen{};
            char *closesthit_closest{};
            char *closesthit_any{};
            char *miss_closest{};
            char *miss_any{};
        };

        class ShaderWrapper : public Noncopyable {
        private:
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
        private:
            DevicePtrTable _device_ptr_table;
            ProgramGroupTable _program_group_table{};
            OptixShaderBindingTable _sbt{};

        public:
            ShaderWrapper(OptixModule optix_module, OptixDeviceContext optix_device_context,
                          const GPUScene *gpu_scene, std::shared_ptr<Device> device,
                          const ProgramName &program_name) {
                create_program_groups(optix_module, optix_device_context, program_name);
                create_sbt(gpu_scene, device);
            }

            void clear() {
                _program_group_table.clear();
                _device_ptr_table = {};
            }

            ~ShaderWrapper() {
                clear();
            }

            void create_program_groups(OptixModule optix_module,
                                       OptixDeviceContext optix_device_context,
                                       const ProgramName &program_name);

            void create_sbt(const GPUScene *gpu_scene, std::shared_ptr<Device> device);
        };
    }
}