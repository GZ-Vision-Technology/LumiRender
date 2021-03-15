//
// Created by Zero on 2021/1/10.
//


#pragma once

#include "cuda_device.h"
#include "graphics/geometry/common.h"
#include "render/include/scene_graph.h"
#include <optix.h>

namespace luminous {
    inline namespace gpu {
        class OptixAccel : public Noncopyable {
        private:
            std::shared_ptr<Device> _device;
            Dispatcher _dispatcher;
            OptixDeviceContext _optix_context{};
            uint32_t geom_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
            size_t _gpu_bvh_bytes = 0;
            std::vector<CUdeviceptr> _vert_buffer_ptr;
            OptixTraversableHandle _root_traversable{};
        private:
            void create_module();

            void init_context();

            OptixBuildInput get_mesh_build_input(const Buffer<float3> &positions,
                                                 const Buffer<TriangleHandle> &triangles,
                                                 const MeshHandle &mesh);

        public:
            OptixAccel(const SP<Device> &device);

            void build_bvh(const Buffer<float3> &positions, const Buffer<TriangleHandle> &triangles,
                           const vector<MeshHandle> &meshes, const Buffer<uint> &instance_list,
                           const vector<float4x4> &transform_list, const vector<uint> &inst_to_transform);
        };
    }
}