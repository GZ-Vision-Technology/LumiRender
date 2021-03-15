//
// Created by Zero on 2021/1/10.
//

#include "optix_accel.h"
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

namespace luminous {
    inline namespace gpu {

        static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */ ) {
            std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
        }

        OptixAccel::OptixAccel(const SP<Device> &device)
                : _device(device),
                  _dispatcher(_device->new_dispatcher()) {
            init_context();
        }

        void OptixAccel::init_context() {
            // Initialize CUDA for this device on this thread
            CU_CHECK(cuMemFree(0));
            OPTIX_CHECK(optixInit());
            OptixDeviceContextOptions ctx_options = {};
#ifndef NDEBUG
            ctx_options.logCallbackLevel = 4; // status/progress
#else
            ctx_options.logCallbackLevel     = 2; // error
#endif
            ctx_options.logCallbackFunction = context_log_cb;
#if (OPTIX_VERSION >= 70200)
            ctx_options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
            // Zero means take the current context
            CUcontext cuda_context = 0;
            OPTIX_CHECK(optixDeviceContextCreate(cuda_context, &ctx_options, &_optix_context));
        }

        void OptixAccel::create_module() {

        }

        OptixBuildInput OptixAccel::get_mesh_build_input(const Buffer<float3> &positions,
                                                         const Buffer<TriangleHandle> &triangles,
                                                         const MeshHandle &mesh) {
            OptixBuildInput input = {};
            input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            {
                input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                input.triangleArray.indexStrideInBytes = sizeof(float3);
                input.triangleArray.numVertices = positions.size();
                _vert_buffer_ptr.push_back(positions.address<CUdeviceptr>(0));
                input.triangleArray.vertexBuffers = &_vert_buffer_ptr.back();
            }
            {
                input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                input.triangleArray.indexStrideInBytes = sizeof(TriangleHandle);
                input.triangleArray.numIndexTriplets = triangles.size();
                input.triangleArray.indexBuffer = triangles.address<CUdeviceptr>(0);
            }
            {
                //todo fix
                input.triangleArray.flags = &geom_flags;
                input.triangleArray.numSbtRecords = 1;
                input.triangleArray.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(nullptr);
                input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
                input.triangleArray.sbtIndexOffsetStrideInBytes = 0;
            }
            return input;
        }

        void OptixAccel::build_bvh(const Buffer<float3> &positions, const Buffer<TriangleHandle> &triangles,
                                   const vector<MeshHandle> &meshes, const Buffer<uint> &instance_list,
                                   const vector<float4x4> &transform_list, const vector<uint> &inst_to_transform) {
            TASK_TAG("build optix bvh");
            vector<OptixBuildInput> inputs;
            for (int i = 0; i < meshes.size(); ++i) {
                if (i == 0 || i == 1)
                inputs.push_back(get_mesh_build_input(positions, triangles, meshes[i]));
            }
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = (OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
            accel_options.motionOptions.numKeys = 1;
            accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

            OptixAccelBufferSizes gas_buffer_sizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(
                    _optix_context,
                    &accel_options,
                    &inputs[0],
                    1,  // num_build_inputs
                    &gas_buffer_sizes
            ));
cout << "adfdsf" << endl;
        }
    }
}