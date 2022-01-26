//
// Created by Zero on 23/08/2021.
//


#pragma once

#include <optix.h>
#include "core/concepts.h"
#include "core/backend/buffer.h"
#include "gpu/framework/cuda_impl.h"
#include "gpu/accel/optix_defines.h"

namespace luminous {

    inline namespace render {
        class Scene;
    }

    inline namespace gpu {

        struct ProgramName {
            const char *raygen{};
            const char *closesthit_closest{};
            const char *closesthit_any{};
            // Direct callable C function names must be terminated with a null pointer.
            const char *const* direct_callables{};
        };

        struct ProgramGroupTable {
            OptixProgramGroup raygen_group{nullptr};
            OptixProgramGroup miss_closest_group{nullptr};
            OptixProgramGroup hit_closest_group{nullptr};
            OptixProgramGroup hit_any_group{nullptr};
            std::vector<OptixProgramGroup> callable_prog_groups;

            ProgramGroupTable() = default;

            static constexpr auto size() {
                return sizeof(ProgramGroupTable) / sizeof(OptixProgramGroup);
            }

            void clear() const {
                OPTIX_CHECK(optixProgramGroupDestroy(raygen_group));
                OPTIX_CHECK(optixProgramGroupDestroy(hit_closest_group));
                OPTIX_CHECK(optixProgramGroupDestroy(hit_any_group));
                for(auto &prog_group : callable_prog_groups) {
                    OPTIX_CHECK(optixProgramGroupDestroy(prog_group));
                }
            }
        };

        class ShaderWrapper : public MovableNonCopyable {
        private:
            Buffer<SBTRecord> _sbt_records{nullptr};
            Buffer<SBTRecord> _callable_records{nullptr};
            OptixShaderBindingTable _sbt{};
            ProgramGroupTable _program_group_table{};
        public:
            ShaderWrapper() = default;

            ShaderWrapper(OptixModule optix_module, OptixDeviceContext optix_device_context,
                          const Scene *scene, Device *device,
                          const ProgramName &program_name);

            LM_NODISCARD const OptixShaderBindingTable *sbt_ptr() const { return &_sbt; }

            LM_NODISCARD std::vector<OptixProgramGroup> program_groups() const {
                std::vector<OptixProgramGroup> all_prog_groups = {_program_group_table.raygen_group,
                        _program_group_table.hit_closest_group,
                        _program_group_table.hit_any_group
                };
                all_prog_groups.insert(all_prog_groups.end(), _program_group_table.callable_prog_groups.begin(),
                                       _program_group_table.callable_prog_groups.end());
                return all_prog_groups;
            };

            void clear() {
                _program_group_table.clear();
                _sbt_records.clear();
                _callable_records.clear();
            }

            ProgramGroupTable create_program_groups(OptixModule optix_module,
                                                    OptixDeviceContext optix_device_context,
                                                    const ProgramName &program_name);

            void build_sbt(const render::Scene *scene, Device *device);
        };
    }
}