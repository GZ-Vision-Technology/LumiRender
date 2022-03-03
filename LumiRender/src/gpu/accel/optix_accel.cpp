//
// Created by Zero on 22/08/2021.
//

#include "optix_accel.h"
#include "util/stats.h"
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <iosfwd>


namespace luminous {
    inline namespace gpu {
        static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */ ) {
            std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
        }

        OptixAccel::OptixAccel(Device *device, Context *context, const Scene *scene)
                : Accelerator(scene),
                  _device(device),
                  _dispatcher(_device->new_dispatcher()),
                  _context(context),
                  _optix_device_context(create_context()) {}

        ShaderWrapper OptixAccel::create_shader_wrapper(const std::string_view &ptx_code, const ProgramName &program_name) {
            auto module = obtain_module(ptx_code);
            return ShaderWrapper{
                    module, _optix_device_context,
                    _scene, _device, program_name
            };
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
                static constexpr uint32_t geom_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
                input.triangleArray.flags = &geom_flags;
                input.triangleArray.numSbtRecords = 1;
                input.triangleArray.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(nullptr);
                input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
                input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
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
            auto tri_gas_buffer = _device->create_buffer(gas_buffer_sizes.outputSizeInBytes);
            auto temp_buffer = _device->create_buffer(gas_buffer_sizes.tempSizeInBytes);
            auto compact_size_buffer = _device->create_buffer<uint64_t>(1);

            OptixAccelEmitDesc emit_desc;
            emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emit_desc.result = compact_size_buffer.ptr<CUdeviceptr>();

            OptixTraversableHandle traversable_handle = 0;
            OPTIX_CHECK(optixAccelBuild(_optix_device_context, nullptr,
                                        &accel_options, &build_input, 1,
                                        temp_buffer.ptr<CUdeviceptr>(), gas_buffer_sizes.tempSizeInBytes,
                                        tri_gas_buffer.ptr<CUdeviceptr>(), gas_buffer_sizes.outputSizeInBytes,
                                        &traversable_handle, &emit_desc, 1));

            auto compacted_gas_size = download<size_t>(emit_desc.result);
            if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
                //todo auto release bug
                auto tri_gas_buffer = _device->create_buffer(compacted_gas_size);
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

        void OptixAccel::build_bvh(const Managed<float3> &positions, const Managed<TriangleHandle> &triangles,
                                   const Managed<MeshHandle> &meshes, const Managed<uint> &instance_list,
                                   const Managed<Transform> &transform_list, const Managed<uint> &inst_to_transform) {
            TASK_TAG("build optix bvh");
            std::list<CUdeviceptr> vert_buffer_ptr;
            vector<OptixTraversableHandle> traversable_handles;
            for (const auto &mesh : meshes) {
                traversable_handles.push_back(build_mesh_bvh(positions.device_buffer(),
                                                             triangles.device_buffer(), mesh, vert_buffer_ptr));
            }

            size_t instance_num = instance_list.size();
            OptixBuildInput instance_input = {};
            _instances = _device->create_buffer<OptixInstance>(instance_num);
            instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            instance_input.instanceArray.numInstances = instance_num;
            instance_input.instanceArray.instances = _instances.ptr<CUdeviceptr>();

            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = (OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
            accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

            OptixAccelBufferSizes ias_buffer_sizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(_optix_device_context,
                                                     &accel_options, &instance_input,
                                                     1,  // num build inputs
                                                     &ias_buffer_sizes));

            auto ias_buffer = _device->create_buffer(ias_buffer_sizes.outputSizeInBytes);
            auto temp_buffer = _device->create_buffer(ias_buffer_sizes.tempSizeInBytes);
            auto compact_size_buffer = _device->create_buffer<uint64_t>(1);
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
            _instances.upload(optix_instances.data());

            OPTIX_CHECK(optixAccelBuild(_optix_device_context,
                                        nullptr, &accel_options,
                                        &instance_input, 1,
                                        temp_buffer.ptr<CUdeviceptr>(),
                                        ias_buffer_sizes.tempSizeInBytes,
                                        ias_buffer.ptr<CUdeviceptr>(),
                                        ias_buffer_sizes.outputSizeInBytes,
                                        &_root_as_handle, &emit_desc, 1));
            auto compacted_gas_size = download<size_t>(emit_desc.result);
            if (compacted_gas_size < ias_buffer_sizes.outputSizeInBytes) {
                ias_buffer = _device->create_buffer(compacted_gas_size);
                OPTIX_CHECK(optixAccelCompact(_optix_device_context, nullptr,
                                              _root_as_handle,
                                              ias_buffer.ptr<CUdeviceptr>(),
                                              compacted_gas_size,
                                              &_root_as_handle));
                _bvh_size_in_bytes += ias_buffer.size_in_bytes();
                _as_buffer_list.push_back(move(ias_buffer));
            } else {
                _bvh_size_in_bytes += ias_buffer.size_in_bytes();
                _as_buffer_list.push_back(move(ias_buffer));
            }
            CU_CHECK(cuCtxSynchronize());
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
            CUcontext cu_context = nullptr;
            OPTIX_CHECK(optixDeviceContextCreate(cu_context, &ctx_options, &_optix_device_context));
            return _optix_device_context;
        }

        OptixModule OptixAccel::create_module(const std::string_view &ptx_code) {
            OptixModule optix_module = nullptr;

            // OptiX module

            OptixModuleCompileOptions module_compile_options = {};
            // TODO: REVIEW THIS
            module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
#ifdef LUMINOUS_NSIGHT_DEBUG
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
#else
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#endif
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

            _pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
            _pipeline_compile_options.usesMotionBlur = false;
            _pipeline_compile_options.numPayloadValues = 2u;
            _pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
            _pipeline_compile_options.numAttributeValues = 2;
            // OPTIX_EXCEPTION_FLAG_NONE;
#ifndef NDEBUG
            _pipeline_compile_options.exceptionFlags =
                    (OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                     OPTIX_EXCEPTION_FLAG_DEBUG);
#else
            _pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
            _pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

            char log[2048];
            size_t log_size = sizeof(log);
            OPTIX_CHECK_WITH_LOG(optixModuleCreateFromPTX(
                    _optix_device_context,
                    &module_compile_options,
                    &_pipeline_compile_options,
                    ptx_code.data(), ptx_code.size(),
                    log, &log_size, &optix_module), log);

            return optix_module;
        }


        void OptixAccel::build_pipeline(const std::vector<OptixProgramGroup> &program_groups) {
            constexpr int max_trace_depth = 2;

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth = max_trace_depth;
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
                    (OptixProgramGroup *) program_groups.data(),
                    program_groups.size(),
                    log, &sizeof_log,
                    &_optix_pipeline), log);

            // Set shaders stack sizes.
            OptixStackSizes stack_sizes = {};
            for(auto &prog_group : program_groups) {
                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes,
                                                   max_trace_depth,
                                                   0, // maxCCDepth
                                                   0, // maxDCDEpth
                                                   &direct_callable_stack_size_from_traversal,
                                                   &direct_callable_stack_size_from_state,
                                                   &continuation_stack_size));
            OPTIX_CHECK(optixPipelineSetStackSize(_optix_pipeline,
                                                  direct_callable_stack_size_from_traversal,
                                                  direct_callable_stack_size_from_state,
                                                  continuation_stack_size,
                                                  2 // maxTraversableDepth
                                                  ));
        }

        void OptixAccel::clear() {
            if (_optix_pipeline) {
                optixPipelineDestroy(_optix_pipeline);
                _optix_pipeline = nullptr;
            }
            clear_modules();
            _as_buffer_list.clear();
            optixDeviceContextDestroy(_optix_device_context);
        }
    }
}