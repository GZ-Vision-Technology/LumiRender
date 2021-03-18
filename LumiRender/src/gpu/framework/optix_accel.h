//
// Created by Zero on 2021/1/10.
//


#pragma once

#include "cuda_device.h"
#include "graphics/geometry/common.h"
#include "render/include/scene_graph.h"
#include <optix.h>
#include "optix_params.h"

namespace luminous {
    inline namespace gpu {


        class OptixAccel : public Noncopyable {
        private:
            std::shared_ptr<Device> _device;
            Dispatcher _dispatcher;
            OptixDeviceContext _optix_device_context{};
            OptixPipeline _optix_pipeline{};
            OptixModule _optix_module{};
            OptixPipelineCompileOptions _pipeline_compile_options = {};
            uint32_t geom_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

            struct ProgramGroupTable {
                OptixProgramGroup raygen_prog_group = 0;
                OptixProgramGroup radiance_miss_group = 0;
                OptixProgramGroup occlusion_miss_group = 0;
                OptixProgramGroup radiance_hit_group = 0;
                OptixProgramGroup occlusion_hit_group = 0;

                constexpr auto size() const {
                    return sizeof(ProgramGroupTable) / sizeof(OptixProgramGroup);
                }
            };

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

            void create_sbt(ProgramGroupTable program_group_table);

            OptixBuildInput get_mesh_build_input(const Buffer<float3> &positions,
                                                 const Buffer<TriangleHandle> &triangles,
                                                 const MeshHandle &mesh,
                                                 std::list<CUdeviceptr> &vert_buffer_ptr);

            OptixTraversableHandle build_mesh_bvh(const Buffer<float3> &positions,
                                                  const Buffer<TriangleHandle> &triangles,
                                                  const MeshHandle &mesh,
                                                  std::list<CUdeviceptr> &_vert_buffer_ptr);

        public:
            OptixAccel(const SP<Device> &device);

            size_t bvh_size_in_bytes() const { return _bvh_size_in_bytes; }

            NDSC std::string description() const;

            void build_bvh(const Buffer<float3> &positions, const Buffer<TriangleHandle> &triangles,
                           const vector<MeshHandle> &meshes, const vector<uint> &instance_list,
                           const vector<float4x4> &transform_list, const vector<uint> &inst_to_transform);
        };
    }
}