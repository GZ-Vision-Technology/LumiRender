//
// Created by Zero on 23/08/2021.
//

#include "shader_wrapper.h"
#include "render/scene/scene.h"
#include <iosfwd>
#include <optix.h>
#include <optix_stubs.h>

namespace luminous {
    inline namespace gpu {

        ShaderWrapper::ShaderWrapper(OptixModule optix_module, OptixDeviceContext optix_device_context,
                                     const Scene *scene, Device *device,
                                     const ProgramName &program_name) {
            _program_group_table = create_program_groups(optix_module, optix_device_context, program_name);
            build_sbt(scene, device);
        }

        ProgramGroupTable ShaderWrapper::create_program_groups(OptixModule optix_module,
                                                               OptixDeviceContext optix_device_context,
                                                               const ProgramName &program_name) {
            OptixProgramGroupOptions program_group_options = {};
            char log[2048];
            size_t sizeof_log = sizeof(log);
            ProgramGroupTable program_group_table;
            {
                OptixProgramGroupDesc raygen_prog_group_desc = {};
                raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                raygen_prog_group_desc.raygen.module = optix_module;
                raygen_prog_group_desc.raygen.entryFunctionName = program_name.raygen;
                OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                        optix_device_context,
                        &raygen_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &(program_group_table.raygen_group)
                ), log);
            }

            {
                OptixProgramGroupDesc hit_prog_group_desc = {};
                hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                hit_prog_group_desc.hitgroup.moduleCH = optix_module;
                hit_prog_group_desc.hitgroup.entryFunctionNameCH = program_name.closesthit_closest;
                sizeof_log = sizeof(log);

                OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                        optix_device_context,
                        &hit_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &(program_group_table.hit_closest_group)
                        ), log);

                memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
                hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                hit_prog_group_desc.hitgroup.moduleCH = optix_module;
                hit_prog_group_desc.hitgroup.entryFunctionNameCH = program_name.closesthit_any;
                sizeof_log = sizeof(log);

                OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                        optix_device_context,
                        &hit_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &(program_group_table.hit_any_group)
                        ), log);
            }

            {
                OptixProgramGroupDesc miss_prog_group_desc = {};
                miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
                sizeof_log = sizeof(log);
                OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                        optix_device_context,
                        &miss_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &(program_group_table.miss_closest_group)
                ), log);
            }

            // Direct callable groups
            if(program_name.direct_callables)
            {
                OptixProgramGroupOptions callable_prog_group_options = {};
                std::vector<OptixProgramGroupDesc> callable_prog_group_descs = {};
                int callable_count = 0, i = 0;
                for (const char *const *callable = program_name.direct_callables; *callable;
                     ++callable, ++callable_count)
                    ;
                if(callable_count != 0) {
                    callable_prog_group_descs.resize(callable_count);

                    for (const char *const *callable = program_name.direct_callables; *callable;
                         ++callable, ++i) {
                        callable_prog_group_descs[i].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
                        callable_prog_group_descs[i].callables.moduleDC = optix_module;
                        callable_prog_group_descs[i].callables.entryFunctionNameDC = *callable;
                        callable_prog_group_descs[i].callables.moduleCC = nullptr;
                        callable_prog_group_descs[i].callables.entryFunctionNameCC = nullptr;
                    }

                    program_group_table.callable_prog_groups.resize(callable_count);

                    OPTIX_CHECK(optixProgramGroupCreate(
                        optix_device_context, callable_prog_group_descs.data(),
                        callable_prog_group_descs.size(), &callable_prog_group_options, log,
                        &sizeof_log,
                        (OptixProgramGroup *)program_group_table.callable_prog_groups.data()));
                }
            }

            return program_group_table;
        }

        void ShaderWrapper::build_sbt(const render::Scene *scene, Device *device) {
            _sbt_records = device->create_buffer<SBTRecord>(4);
            SBTRecord sbt[4] = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(_program_group_table.raygen_group,
                                                 &sbt[0]));
            OPTIX_CHECK(optixSbtRecordPackHeader(_program_group_table.hit_closest_group,
                                                 &sbt[1]));
            OPTIX_CHECK(optixSbtRecordPackHeader(_program_group_table.hit_any_group,
                                                 &sbt[2]));
            OPTIX_CHECK(optixSbtRecordPackHeader(_program_group_table.miss_closest_group,
                                                 &sbt[3]));
            _sbt_records.upload(sbt);

            _sbt.raygenRecord = _sbt_records.ptr<CUdeviceptr>();
            _sbt.hitgroupRecordBase = _sbt_records.address<CUdeviceptr>(1);
            _sbt.hitgroupRecordStrideInBytes = sizeof(SBTRecord);
            _sbt.hitgroupRecordCount = 2;
            _sbt.missRecordBase = _sbt_records.address<CUdeviceptr>(3);
            _sbt.missRecordStrideInBytes = sizeof(SBTRecord);
            _sbt.missRecordCount = 1;

            if(!_program_group_table.callable_prog_groups.empty()) {
                _callable_records = device->create_buffer<SBTRecord>(_program_group_table.callable_prog_groups.size());
                std::vector<SBTRecord> callables(_program_group_table.callable_prog_groups.size());
                int i = 0;
                for(auto &prog_group : _program_group_table.callable_prog_groups) {
                    OPTIX_CHECK(optixSbtRecordPackHeader(prog_group, &callables[i]));
                    ++i;
                }
                _callable_records.upload(callables.data(), callables.size());

                _sbt.callablesRecordBase = _callable_records.ptr<CUdeviceptr>();
                _sbt.callablesRecordCount = _callable_records.size();
                _sbt.callablesRecordStrideInBytes = sizeof(SBTRecord);
            }
        }

    }
}