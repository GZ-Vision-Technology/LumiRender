//
// Created by Zero on 2021/1/10.
//

#include "optix_accel.h"
#include <optix_function_table_definition.h>
#include "../gpu_scene.h"
#include "render/include/shader_data.h"
#include "util/stats.h"

extern "C" char optix_shader_code[];

extern "C" char sdk_ptx[];

namespace luminous {
    inline namespace gpu {

        static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */ ) {
            std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
        }

        OptixAccel::OptixAccel(const SP<Device> &device, const GPUScene *gpu_scene)
                : _device(device),
                  _dispatcher(_device->new_dispatcher()) {
            _optix_device_context = create_context();
            _optix_module = create_module(_optix_device_context);
            _program_group_table = create_program_groups(_optix_module);
            _optix_pipeline = create_pipeline(_program_group_table);
            create_sbt(_program_group_table, gpu_scene);
        }

        OptixDeviceContext OptixAccel::create_context() {
            // Initialize CUDA for this device on this thread
            CU_CHECK(cuMemFree(0));
            OPTIX_CHECK(optixInit());
            OptixDeviceContextOptions ctx_options = {};
#ifndef NDEBUG
            ctx_options.logCallbackLevel = 4; // status/progress
#else
            ctx_options.logCallbackLevel = 2; // error
#endif
            ctx_options.logCallbackFunction = context_log_cb;
#if (OPTIX_VERSION >= 70200)
            ctx_options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
#endif
            // Zero means take the current context
            CUcontext cu_context = 0;
            OPTIX_CHECK(optixDeviceContextCreate(cu_context, &ctx_options, &_optix_device_context));
            return _optix_device_context;
        }

        OptixModule OptixAccel::create_module(OptixDeviceContext optix_device_context) {
            OptixModule optix_module = 0;

            // OptiX module
            OptixModuleCompileOptions module_compile_options = {};
            // TODO: REVIEW THIS
            module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

            _pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
            _pipeline_compile_options.usesMotionBlur = false;
            _pipeline_compile_options.numPayloadValues = 2;
            _pipeline_compile_options.numAttributeValues = 2;
            // OPTIX_EXCEPTION_FLAG_NONE;
//#ifndef NDEBUG
//            _pipeline_compile_options.exceptionFlags =
//                    (OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
//                     OPTIX_EXCEPTION_FLAG_DEBUG);
//#else
            _pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
//#endif
            _pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

            char log[2048];
            size_t log_size = sizeof(log);
            std::string ptx_code(optix_shader_code);
            OPTIX_CHECK_WITH_LOG(optixModuleCreateFromPTX(
                    _optix_device_context,
                    &module_compile_options,
                    &_pipeline_compile_options,
                    ptx_code.c_str(), ptx_code.size(),
                    log, &log_size, &optix_module), log);

            return optix_module;
        }

        OptixAccel::ProgramGroupTable OptixAccel::create_program_groups(OptixModule optix_module) {
            ProgramGroupTable program_group_table;
            OptixProgramGroupOptions program_group_options = {};
            char log[2048];
            size_t sizeof_log = sizeof(log);

            {
                OptixProgramGroupDesc raygen_prog_group_desc = {};
                raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                raygen_prog_group_desc.raygen.module = _optix_module;
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
                miss_prog_group_desc.miss.module = _optix_module;
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
                miss_prog_group_desc.miss.module = _optix_module;  // NULL miss program for occlusion rays
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
                hit_prog_group_desc.hitgroup.moduleCH = _optix_module;
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
                hit_prog_group_desc.hitgroup.moduleCH = _optix_module;
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

        OptixPipeline OptixAccel::create_pipeline(OptixAccel::ProgramGroupTable program_group_table) {
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

        void OptixAccel::create_sbt(ProgramGroupTable program_group_table, const GPUScene *gpu_scene) {

            auto fill_group_data = [&](HitGroupRecord *p, const GPUScene *gpu_scene) {
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

            _device_ptr_table.miss_record = _device->allocate_buffer<MissRecord>(RayType::Count);
            MissRecord ms_sbt[RayType::Count] = {};
            ms_sbt[RayType::Radiance].data.light_sampler = gpu_scene->_light_sampler.device_data();
            ms_sbt[RayType::Occlusion].data.light_sampler = gpu_scene->_light_sampler.device_data();
            ms_sbt[RayType::Radiance].data.textures = gpu_scene->_textures.device_buffer_view();
            ms_sbt[RayType::Occlusion].data.textures = gpu_scene->_textures.device_buffer_view();
            OPTIX_CHECK(optixSbtRecordPackHeader(_program_group_table.radiance_miss_group, &ms_sbt[RayType::Radiance]));
            OPTIX_CHECK(
                    optixSbtRecordPackHeader(_program_group_table.occlusion_miss_group, &ms_sbt[RayType::Occlusion]));
            _device_ptr_table.miss_record.upload(ms_sbt);

            _device_ptr_table.hit_record = _device->allocate_buffer<HitGroupRecord>(RayType::Count);
            HitGroupRecord hit_sbt[RayType::Count] = {};
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

        OptixBuildInput OptixAccel::get_mesh_build_input(const Buffer<const float3> &positions,
                                                         const Buffer<const TriangleHandle> &triangles,
                                                         const MeshHandle &mesh,
                                                         std::list<CUdeviceptr> &vert_buffer_ptr) {
            OptixBuildInput input = {};
            input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            {
                input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                input.triangleArray.vertexStrideInBytes = sizeof(float3);
                input.triangleArray.numVertices = mesh.vertex_count;
                vert_buffer_ptr.push_back(positions.address<CUdeviceptr>(mesh.vertex_offset));
                input.triangleArray.vertexBuffers = &vert_buffer_ptr.back();
            }
            {
                input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                input.triangleArray.indexStrideInBytes = sizeof(TriangleHandle);
                input.triangleArray.numIndexTriplets = mesh.triangle_count;
                input.triangleArray.indexBuffer = triangles.address<CUdeviceptr>(mesh.triangle_offset);
            }
            {
                //todo fix
                input.triangleArray.flags = &geom_flags;
                input.triangleArray.numSbtRecords = 1;
                input.triangleArray.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(nullptr);
                input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof( uint32_t );
                input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );
            }
            return input;
        }

        OptixTraversableHandle OptixAccel::build_mesh_bvh(const Buffer<const float3> &positions,
                                                          const Buffer<const TriangleHandle> &triangles,
                                                          const MeshHandle &mesh,
                                                          std::list<CUdeviceptr> &vert_buffer_ptr) {

            OptixBuildInput build_input = get_mesh_build_input(positions, triangles, mesh, vert_buffer_ptr);

            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = (OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
            accel_options.motionOptions.numKeys = 1;
            accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

            OptixAccelBufferSizes gas_buffer_sizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(
                    _optix_device_context,
                    &accel_options,
                    &build_input,
                    1,  // num_build_inputs
                    &gas_buffer_sizes
            ));
            auto tri_gas_buffer = _device->allocate_buffer(gas_buffer_sizes.outputSizeInBytes);
            auto temp_buffer = _device->allocate_buffer(gas_buffer_sizes.tempSizeInBytes);
            auto compact_size_buffer = _device->allocate_buffer<uint64_t>(1);

            OptixAccelEmitDesc emit_desc;
            emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emit_desc.result = compact_size_buffer.ptr<CUdeviceptr>();

            OptixTraversableHandle traversable_handle = 0;
            OPTIX_CHECK(optixAccelBuild(_optix_device_context, 0, &accel_options,
                                        &build_input, 1,
                                        temp_buffer.ptr<CUdeviceptr>(), gas_buffer_sizes.tempSizeInBytes,
                                        tri_gas_buffer.ptr<CUdeviceptr>(), gas_buffer_sizes.outputSizeInBytes,
                                        &traversable_handle, &emit_desc, 1));

            auto compacted_gas_size = download<size_t>(emit_desc.result);
            if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
                //todo auto release bug
                auto tri_gas_buffer = _device->allocate_buffer(compacted_gas_size);
                OPTIX_CHECK(optixAccelCompact(_optix_device_context, nullptr,
                                              traversable_handle,
                                              tri_gas_buffer.ptr<CUdeviceptr>(),
                                              compacted_gas_size,
                                              &traversable_handle));
                _bvh_size_in_bytes += tri_gas_buffer.size_in_bytes();
                _as_buffer_list.push_back(move(tri_gas_buffer));
            } else {
                _bvh_size_in_bytes += tri_gas_buffer.size_in_bytes();
                _as_buffer_list.push_back(move(tri_gas_buffer));
            }
            CU_CHECK(cuCtxSynchronize());
            return traversable_handle;
        }

        void OptixAccel::build_bvh(const Buffer<const float3> &positions, const Buffer<const TriangleHandle> &triangles,
                                   const Managed<MeshHandle> &meshes, const Managed<uint> &instance_list,
                                   const Managed<Transform> &transform_list, const Managed<uint> &inst_to_transform) {
            TASK_TAG("build optix bvh");
            std::list<CUdeviceptr> vert_buffer_ptr;
            vector<OptixTraversableHandle> traversable_handles;
            for (const auto &mesh : meshes) {
                traversable_handles.push_back(build_mesh_bvh(positions, triangles, mesh, vert_buffer_ptr));
            }

            size_t instance_num = instance_list.size();
            OptixBuildInput instance_input = {};
            _device_ptr_table.instances = _device->allocate_buffer<OptixInstance>(instance_num);
            instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            instance_input.instanceArray.numInstances = instance_num;
            instance_input.instanceArray.instances = _device_ptr_table.instances.ptr<CUdeviceptr>();

            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = (OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
            accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

            OptixAccelBufferSizes ias_buffer_sizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(_optix_device_context,
                                                     &accel_options, &instance_input,
                                                     1,  // num build inputs
                                                     &ias_buffer_sizes));

            auto ias_buffer = _device->allocate_buffer(ias_buffer_sizes.outputSizeInBytes);
            auto temp_buffer = _device->allocate_buffer(ias_buffer_sizes.tempSizeInBytes);
            auto compact_size_buffer = _device->allocate_buffer<uint64_t>(1);
            OptixAccelEmitDesc emit_desc;
            emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emit_desc.result = compact_size_buffer.ptr<uint64_t>();


            vector<OptixInstance> optix_instances;
            optix_instances.reserve(instance_num);
            for (int i = 0; i < instance_list.size(); ++i) {
                uint mesh_idx = instance_list[i];
                uint transform_idx = inst_to_transform[i];
                Transform transform = transform_list[transform_idx];
                OptixInstance optix_instance;
                optix_instance.traversableHandle = traversable_handles[mesh_idx];
                optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                optix_instance.instanceId = i;
                optix_instance.visibilityMask = 1;
                optix_instance.sbtOffset = 0;
                mat4x4_to_array12(transform.mat4x4(), optix_instance.transform);
                optix_instances.push_back(optix_instance);
            }
            _device_ptr_table.instances.upload(optix_instances.data());

            OPTIX_CHECK(optixAccelBuild(_optix_device_context,
                                        0, &accel_options,
                                        &instance_input, 1,
                                        temp_buffer.ptr<CUdeviceptr>(),
                                        ias_buffer_sizes.tempSizeInBytes,
                                        ias_buffer.ptr<CUdeviceptr>(),
                                        ias_buffer_sizes.outputSizeInBytes,
                                        &_root_ias_handle, &emit_desc, 1));
            auto compacted_gas_size = download<size_t>(emit_desc.result);
            if (compacted_gas_size < ias_buffer_sizes.outputSizeInBytes) {
                ias_buffer = _device->allocate_buffer(compacted_gas_size);
                OPTIX_CHECK(optixAccelCompact(_optix_device_context, nullptr,
                                              _root_ias_handle,
                                              ias_buffer.ptr<CUdeviceptr>(),
                                              compacted_gas_size,
                                              &_root_ias_handle));
                _bvh_size_in_bytes += ias_buffer.size_in_bytes();
                _as_buffer_list.push_back(move(ias_buffer));
            } else {
                _bvh_size_in_bytes += ias_buffer.size_in_bytes();
                _as_buffer_list.push_back(move(ias_buffer));
            }
            CU_CHECK(cuCtxSynchronize());
//            _path_tracer_state.device = _device;
//            init(_path_tracer_state);
//            _path_tracer_state.stream = dynamic_cast<CUDADispatcher *>(_dispatcher.impl_mut())->stream;
        }

        void OptixAccel::launch(uint2 res, Managed<LaunchParams> &launch_params) {
//            _path_tracer_state.params = launch_params.front();
//            _path_tracer_state.params.traversable_handle = _path_tracer_state.ias_handle;
//            launchSubframe(_path_tracer_state);
//            return;
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

        void OptixAccel::clear() {
            optixPipelineDestroy(_optix_pipeline);
            _device_ptr_table = {};
            _program_group_table.clear();
            _as_buffer_list.clear();
            optixModuleDestroy(_optix_module);
            optixDeviceContextDestroy(_optix_device_context);
        }

        std::string OptixAccel::description() const {
            float size_in_M = (_bvh_size_in_bytes * 1.f) / (sqr(1024));
            return string_printf("bvh size is %f MB\n", size_in_M);
        }

        OptixAccel::~OptixAccel() {
            clear();
        }

        void OptixAccel::test() {

        }
    }
}