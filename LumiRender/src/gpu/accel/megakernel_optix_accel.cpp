//
// Created by Zero on 2021/1/10.
//

#include "megakernel_optix_accel.h"
#include <optix_function_table_definition.h>
#include "gpu/gpu_scene.h"
#include <iosfwd>

extern "C" char optix_shader_code[];

namespace luminous {
    inline namespace gpu {

        MegakernelOptixAccel::MegakernelOptixAccel(const SP<Device> &device, const GPUScene *gpu_scene, Context *context)
                : OptixAccel(device, context){
            _program_group_table = create_program_groups(obtain_module(optix_shader_code));
            _optix_pipeline = create_pipeline();
            create_sbt(_program_group_table, gpu_scene);
        }

        MegakernelOptixAccel::ProgramGroupTable MegakernelOptixAccel::create_program_groups(OptixModule optix_module) {
            ProgramGroupTable program_group_table;
            OptixProgramGroupOptions program_group_options = {};
            char log[2048];
            size_t sizeof_log = sizeof(log);

            {
                OptixProgramGroupDesc raygen_prog_group_desc = {};
                raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                raygen_prog_group_desc.raygen.module = optix_module;
                raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
                OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                        _optix_device_context,
                        &raygen_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &(program_group_table.raygen_prog_group)
                ), log);
            }

            {
                OptixProgramGroupDesc miss_prog_group_desc = {};
                miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
                miss_prog_group_desc.miss.module = optix_module;
                miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
                sizeof_log = sizeof(log);
                OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                        _optix_device_context,
                        &miss_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &(program_group_table.radiance_miss_group)
                ), log);

                memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
                miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
                miss_prog_group_desc.miss.module = optix_module;  // NULL miss program for occlusion rays
                miss_prog_group_desc.miss.entryFunctionName = "__miss__shadow";
                sizeof_log = sizeof(log);

                OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                        _optix_device_context,
                        &miss_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &(program_group_table.occlusion_miss_group)
                ), log);
            }

            {
                OptixProgramGroupDesc hit_prog_group_desc = {};
                hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                hit_prog_group_desc.hitgroup.moduleCH = optix_module;
                hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
                sizeof_log = sizeof(log);

                OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                        _optix_device_context,
                        &hit_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &(program_group_table.radiance_hit_group)
                ), log);

                memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
                hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                hit_prog_group_desc.hitgroup.moduleCH = optix_module;
                hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
                sizeof_log = sizeof(log);

                OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                        _optix_device_context,
                        &hit_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &(program_group_table.occlusion_hit_group)
                ), log);
            }

            return program_group_table;
        }

        OptixPipeline MegakernelOptixAccel::create_pipeline() {
            OptixPipeline pipeline = 0;
            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth = 2;
#ifndef NDEBUG
            pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
            pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
            char log[2048];
            size_t sizeof_log = sizeof(log);

            OPTIX_CHECK_WITH_LOG(optixPipelineCreate(
                    _optix_device_context,
                    &_pipeline_compile_options,
                    &pipeline_link_options,
                    (OptixProgramGroup *) &_program_group_table,
                    _program_group_table.size(),
                    log, &sizeof_log,
                    &pipeline
            ), log);

            return pipeline;
        }

        void MegakernelOptixAccel::create_sbt(ProgramGroupTable program_group_table, const GPUScene *gpu_scene) {

            auto fill_group_data = [&](SceneRecord *p, const GPUScene *gpu_scene) {
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

            _device_ptr_table.rg_record = _device->allocate_buffer<RayGenRecord>(1);
            RayGenRecord rg_sbt = {};
            OPTIX_CHECK(optixSbtRecordPackHeader(_program_group_table.raygen_prog_group, &rg_sbt));
            _device_ptr_table.rg_record.upload(&rg_sbt);

            _device_ptr_table.miss_record = _device->allocate_buffer<SceneRecord>(RayType::Count);
            SceneRecord ms_sbt[RayType::Count] = {};
            fill_group_data(&ms_sbt[RayType::Radiance], gpu_scene);
            fill_group_data(&ms_sbt[RayType::Occlusion], gpu_scene);
            OPTIX_CHECK(optixSbtRecordPackHeader(_program_group_table.radiance_miss_group, &ms_sbt[RayType::Radiance]));
            OPTIX_CHECK(
                    optixSbtRecordPackHeader(_program_group_table.occlusion_miss_group, &ms_sbt[RayType::Occlusion]));
            _device_ptr_table.miss_record.upload(ms_sbt);

            _device_ptr_table.hit_record = _device->allocate_buffer<SceneRecord>(RayType::Count);
            SceneRecord hit_sbt[RayType::Count] = {};
            fill_group_data(&hit_sbt[RayType::Radiance], gpu_scene);
            OPTIX_CHECK(optixSbtRecordPackHeader(_program_group_table.radiance_hit_group,
                                                 &hit_sbt[RayType::Radiance]));
            fill_group_data(&hit_sbt[RayType::Occlusion], gpu_scene);
            OPTIX_CHECK(optixSbtRecordPackHeader(_program_group_table.occlusion_hit_group,
                                                 &hit_sbt[RayType::Occlusion]));
            _device_ptr_table.hit_record.upload(hit_sbt);

            _sbt.raygenRecord = _device_ptr_table.rg_record.ptr<CUdeviceptr>();
            _sbt.missRecordBase = _device_ptr_table.miss_record.ptr<CUdeviceptr>();
            _sbt.missRecordStrideInBytes = _device_ptr_table.miss_record.stride_in_bytes();
            _sbt.missRecordCount = _device_ptr_table.miss_record.size();
            _sbt.hitgroupRecordBase = _device_ptr_table.hit_record.ptr<CUdeviceptr>();
            _sbt.hitgroupRecordStrideInBytes = _device_ptr_table.hit_record.stride_in_bytes();
            _sbt.hitgroupRecordCount = _device_ptr_table.hit_record.size();
        }

        void MegakernelOptixAccel::launch(uint2 res, Managed<LaunchParams> &launch_params) {
            auto stream = dynamic_cast<CUDADispatcher *>(_dispatcher.impl_mut())->stream;
            auto x = res.x;
            auto y = res.y;
            launch_params->traversable_handle = _root_ias_handle;
            launch_params.synchronize_to_gpu();
            OPTIX_CHECK(optixLaunch(_optix_pipeline,
                                    stream,
                                    launch_params.device_ptr<CUdeviceptr>(),
                                    sizeof(LaunchParams),
                                    &_sbt,
                                    x,
                                    y,
                                    1u));
            _dispatcher.wait();
        }

        void MegakernelOptixAccel::clear() {
            optixPipelineDestroy(_optix_pipeline);
            _program_group_table.clear();
            _device_ptr_table = {};
            OptixAccel::clear();
        }

        MegakernelOptixAccel::~MegakernelOptixAccel() {
            clear();
        }
    }
}