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
                input.triangleArray.numVertices = mesh.vertex_count;
                _vert_buffer_ptr.push_back(positions.address<CUdeviceptr>(mesh.vertex_offset));
                input.triangleArray.vertexBuffers = &_vert_buffer_ptr.back();
            }
            {
                input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                input.triangleArray.indexStrideInBytes = sizeof(TriangleHandle);
                input.triangleArray.numIndexTriplets = triangles.size();
                input.triangleArray.indexBuffer = triangles.address<CUdeviceptr>(mesh.triangle_offset);
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

        void OptixAccel::build_bvh(const Buffer<float3> &positions, const Buffer<uint> &triangles,
                                   const vector<MeshHandle> &meshes, const Buffer<uint> &instance_list,
                                   const vector<uint> &transform_list, const vector<uint> &inst_to_transform) {

        }
    }
}