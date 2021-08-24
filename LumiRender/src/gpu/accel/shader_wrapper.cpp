//
// Created by Zero on 23/08/2021.
//

#include "shader_wrapper.h"
#include "gpu/gpu_scene.h"
#include <iosfwd>

namespace luminous {
    inline namespace gpu {
        void ShaderWrapper::create_program_groups(OptixModule optix_module,
                                                  OptixDeviceContext optix_device_context,
                                                  const ProgramName &program_name) {
            OptixProgramGroupOptions program_group_options = {};
            char log[2048];
            size_t sizeof_log = sizeof(log);

            {
                OptixProgramGroupDesc raygen_prog_group_desc = {};
                raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                raygen_prog_group_desc.raygen.module = optix_module;
                raygen_prog_group_desc.raygen.entryFunctionName = program_name.raygen;
                raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
                OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                        optix_device_context,
                        &raygen_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &(_program_group_table.raygen_prog_group)
                        ), log);
            }

            {
                OptixProgramGroupDesc miss_prog_group_desc = {};
                miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
                miss_prog_group_desc.miss.module = optix_module;
                miss_prog_group_desc.miss.entryFunctionName = program_name.miss_closest;
                miss_prog_group_desc.miss.entryFunctionName = "__miss__closest";
                sizeof_log = sizeof(log);
                OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                        optix_device_context,
                        &miss_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &(_program_group_table.radiance_miss_group)
                        ), log);

                memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
                miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
                miss_prog_group_desc.miss.module = optix_module;  // NULL miss program for occlusion rays
                miss_prog_group_desc.miss.entryFunctionName = program_name.miss_any;
                miss_prog_group_desc.miss.entryFunctionName = "__miss__any";
                sizeof_log = sizeof(log);

                OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                        optix_device_context,
                        &miss_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &(_program_group_table.occlusion_miss_group)
                        ), log);
            }

            {
                OptixProgramGroupDesc hit_prog_group_desc = {};
                hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                hit_prog_group_desc.hitgroup.moduleCH = optix_module;
                hit_prog_group_desc.hitgroup.entryFunctionNameCH = program_name.closesthit_closest;
                hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__closest";
                sizeof_log = sizeof(log);

                OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                        optix_device_context,
                        &hit_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &(_program_group_table.radiance_hit_group)
                        ), log);

                memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
                hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                hit_prog_group_desc.hitgroup.moduleCH = optix_module;
                hit_prog_group_desc.hitgroup.entryFunctionNameCH = program_name.closesthit_any;
                hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__any";
                sizeof_log = sizeof(log);

                OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                        optix_device_context,
                        &hit_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &(_program_group_table.occlusion_hit_group)
                        ), log);
            }
        }

        void ShaderWrapper::create_sbt(const GPUScene *gpu_scene, std::shared_ptr<Device> device) {
            auto fill_scene_data = [&](SceneRecord *p, const GPUScene *gpu_scene) {
                p->data.positions = gpu_scene->_positions.device_buffer_view();
                p->data.normals = gpu_scene->_normals.device_buffer_view();
                p->data.tex_coords = gpu_scene->_tex_coords.device_buffer_view();
                p->data.triangles = gpu_scene->_triangles.device_buffer_view();
                p->data.meshes = gpu_scene->_meshes.device_buffer_view();

                p->data.inst_to_mesh_idx = gpu_scene->_inst_to_mesh_idx.device_buffer_view();
                p->data.inst_to_transform_idx = gpu_scene->_inst_to_transform_idx.device_buffer_view();
                p->data.transforms = gpu_scene->_transforms.device_buffer_view();

                p->data.light_sampler = gpu_scene->_light_sampler.device_data();
                p->data.distributions = gpu_scene->_distribution_mgr.distributions.device_buffer_view();
                p->data.distribution2ds = gpu_scene->_distribution_mgr.distribution2ds.device_buffer_view();

                p->data.textures = gpu_scene->_textures.device_buffer_view();
                p->data.materials = gpu_scene->_materials.device_buffer_view();
            };

            _device_ptr_table.rg_record = device->allocate_buffer<RayGenRecord>(1);
            RayGenRecord rg_sbt = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(_program_group_table.raygen_prog_group, &rg_sbt));
            _device_ptr_table.rg_record.upload(&rg_sbt);

            _device_ptr_table.miss_record = device->allocate_buffer<SceneRecord>(RayType::Count);
            SceneRecord ms_sbt[RayType::Count] = {};
            fill_scene_data(&ms_sbt[RayType::ClosestHit], gpu_scene);
            fill_scene_data(&ms_sbt[RayType::AnyHit], gpu_scene);
            OPTIX_CHECK(
                    optixSbtRecordPackHeader(_program_group_table.radiance_miss_group, &ms_sbt[RayType::ClosestHit]));
            OPTIX_CHECK(
                    optixSbtRecordPackHeader(_program_group_table.occlusion_miss_group, &ms_sbt[RayType::AnyHit]));
            _device_ptr_table.miss_record.upload(ms_sbt);

            _device_ptr_table.hit_record = device->allocate_buffer<SceneRecord>(RayType::Count);
            SceneRecord hit_sbt[RayType::Count] = {};
            fill_scene_data(&hit_sbt[RayType::ClosestHit], gpu_scene);
            OPTIX_CHECK(optixSbtRecordPackHeader(_program_group_table.radiance_hit_group,
                                                 &hit_sbt[RayType::ClosestHit]));
            fill_scene_data(&hit_sbt[RayType::AnyHit], gpu_scene);
            OPTIX_CHECK(optixSbtRecordPackHeader(_program_group_table.occlusion_hit_group,
                                                 &hit_sbt[RayType::AnyHit]));
            _device_ptr_table.hit_record.upload(hit_sbt);

            _sbt.raygenRecord = _device_ptr_table.rg_record.ptr<CUdeviceptr>();
            _sbt.missRecordBase = _device_ptr_table.miss_record.ptr<CUdeviceptr>();
            _sbt.missRecordStrideInBytes = _device_ptr_table.miss_record.stride_in_bytes();
            _sbt.missRecordCount = _device_ptr_table.miss_record.size();
            _sbt.hitgroupRecordBase = _device_ptr_table.hit_record.ptr<CUdeviceptr>();
            _sbt.hitgroupRecordStrideInBytes = _device_ptr_table.hit_record.stride_in_bytes();
            _sbt.hitgroupRecordCount = _device_ptr_table.hit_record.size();
        }

    }
}