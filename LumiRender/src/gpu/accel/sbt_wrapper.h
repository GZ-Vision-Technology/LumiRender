//
// Created by Zero on 23/08/2021.
//


#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include "core/concepts.h"
#include "core/backend/buffer.h"

namespace luminous {
    inline namespace gpu {
        class RayGenRecord;

        class SceneRecord;

        class SbtWrapper : public Noncopyable {
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
            SbtWrapper(OptixModule optix_module) {

            }
        };
    }
}