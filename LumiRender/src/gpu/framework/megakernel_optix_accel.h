//
// Created by Zero on 2021/1/10.
//


#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include "cuda_impl.h"
#include "base_libs/geometry/common.h"
#include "render/include/scene_graph.h"
#include "optix_params.h"
#include "core/backend/managed.h"
#include "gpu/framework/optix_accel.h"

namespace luminous {
    inline namespace gpu {

        class GPUScene;

        class MegakernelOptixAccel : public Noncopyable {
        private:
            Context *_context{};
            std::shared_ptr<Device> _device;
            Dispatcher _dispatcher;
            OptixDeviceContext _optix_device_context{};
            OptixPipeline _optix_pipeline{};
            OptixModule _optix_module{};
            OptixPipelineCompileOptions _pipeline_compile_options = {};
            uint32_t geom_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

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
                Buffer<OptixInstance> instances{nullptr};
            };

            DevicePtrTable _device_ptr_table;

            ProgramGroupTable _program_group_table{};

            OptixShaderBindingTable _sbt{};
            OptixTraversableHandle _root_ias_handle{};

            size_t _bvh_size_in_bytes{0u};

            std::list<Buffer<std::byte>> _as_buffer_list;
        private:

            OptixDeviceContext create_context();

            OptixModule create_module(OptixDeviceContext optix_device_context);

            ProgramGroupTable create_program_groups(OptixModule optix_module);

            OptixPipeline create_pipeline(ProgramGroupTable program_group_table);

            void create_sbt(ProgramGroupTable program_group_table, const GPUScene *gpu_scene);

            OptixBuildInput get_mesh_build_input(const Buffer<const float3> &positions,
                                                 const Buffer<const TriangleHandle> &triangles,
                                                 const MeshHandle &mesh,
                                                 std::list<CUdeviceptr> &vert_buffer_ptr);

            OptixTraversableHandle build_mesh_bvh(const Buffer<const float3> &positions,
                                                  const Buffer<const TriangleHandle> &triangles,
                                                  const MeshHandle &mesh,
                                                  std::list<CUdeviceptr> &_vert_buffer_ptr);

        public:
            MegakernelOptixAccel(const SP<Device> &device, const GPUScene *gpu_scene, Context *context);

            ~MegakernelOptixAccel();

            NDSC size_t bvh_size_in_bytes() const { return _bvh_size_in_bytes; }

            void launch(uint2 res, Managed<LaunchParams> &launch_params);

            void clear();

            NDSC std::string description() const;

            void build_bvh(const Buffer<const float3> &positions, const Buffer<const TriangleHandle> &triangles,
                           const Managed<MeshHandle> &meshes, const Managed<uint> &instance_list,
                           const Managed<Transform> &transform_list, const Managed<uint> &inst_to_transform);
        };
    }
}