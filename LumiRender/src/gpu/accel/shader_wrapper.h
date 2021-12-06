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
            const char *miss_closest{};
            const char *miss_any{};
        };

        struct ProgramGroupTable {
            OptixProgramGroup raygen_group{nullptr};
            OptixProgramGroup miss_closest_group{nullptr};
            OptixProgramGroup miss_any_group{nullptr};
            OptixProgramGroup hit_closest_group{nullptr};
            OptixProgramGroup hit_any_group{nullptr};

            ProgramGroupTable() = default;

            static constexpr auto size() {
                return sizeof(ProgramGroupTable) / sizeof(OptixProgramGroup);
            }

            void clear() const {
                OPTIX_CHECK(optixProgramGroupDestroy(raygen_group));
                OPTIX_CHECK(optixProgramGroupDestroy(miss_closest_group));
                OPTIX_CHECK(optixProgramGroupDestroy(miss_any_group));
                OPTIX_CHECK(optixProgramGroupDestroy(hit_closest_group));
                OPTIX_CHECK(optixProgramGroupDestroy(hit_any_group));
            }
        };

        struct DevicePtrTable {
            Buffer<SBTRecord> rg_record{nullptr};
            Buffer<SBTRecord> miss_record{nullptr};
            Buffer<SBTRecord> hit_record{nullptr};

            void clear() {
                rg_record.clear();
                miss_record.clear();
                hit_record.clear();
            }
        };

        class ShaderWrapper : public Noncopyable {
        private:
            DevicePtrTable _device_ptr_table;
            OptixShaderBindingTable _sbt{};
            ProgramGroupTable _program_group_table{};
        public:
            ShaderWrapper() = default;

            ShaderWrapper(ShaderWrapper &&other) noexcept
                    : _device_ptr_table(std::move(other._device_ptr_table)),
                      _sbt(other._sbt),
                      _program_group_table(other._program_group_table) {}

            ShaderWrapper(OptixModule optix_module, OptixDeviceContext optix_device_context,
                          const Scene *scene, Device *device,
                          const ProgramName &program_name);

            LM_NODISCARD const OptixShaderBindingTable *sbt_ptr() const { return &_sbt; }

            LM_NODISCARD std::vector<OptixProgramGroup> program_groups() const {
                return {_program_group_table.raygen_group,
                        _program_group_table.miss_closest_group,
                        _program_group_table.miss_any_group,
                        _program_group_table.hit_closest_group,
                        _program_group_table.hit_any_group};
            };

            void clear() {
                _program_group_table.clear();
                _device_ptr_table.clear();
            }

            ~ShaderWrapper() = default;

            ProgramGroupTable create_program_groups(OptixModule optix_module,
                                                    OptixDeviceContext optix_device_context,
                                                    const ProgramName &program_name);

            void build_sbt(const render::Scene *scene, Device *device);
        };
    }
}