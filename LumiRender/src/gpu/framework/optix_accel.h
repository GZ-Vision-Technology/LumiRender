//
// Created by Zero on 22/08/2021.
//


#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include "cuda_impl.h"
#include "base_libs/geometry/common.h"
#include "render/include/scene_graph.h"
#include "optix_params.h"
#include "core/backend/managed.h"

namespace luminous {
    inline namespace gpu {

        class GPUScene;

        class OptixAccel : public Noncopyable {
        protected:
            OptixDeviceContext _optix_device_context{};
            std::shared_ptr<Device> _device;
            Dispatcher _dispatcher;
            uint32_t geom_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
            OptixTraversableHandle _root_ias_handle{};
            size_t _bvh_size_in_bytes{0u};
            std::list<Buffer<std::byte>> _as_buffer_list;
            Buffer<OptixInstance> _instances{nullptr};

            OptixBuildInput get_mesh_build_input(const Buffer<const float3> &positions,
                                                 const Buffer<const TriangleHandle> &triangles,
                                                 const MeshHandle &mesh,
                                                 std::list<CUdeviceptr> &vert_buffer_ptr);

            OptixTraversableHandle build_mesh_bvh(const Buffer<const float3> &positions,
                                                  const Buffer<const TriangleHandle> &triangles,
                                                  const MeshHandle &mesh,
                                                  std::list<CUdeviceptr> &_vert_buffer_ptr);

        public:
            explicit OptixAccel(const SP<Device> &device);

            OptixDeviceContext create_context();

            NDSC size_t bvh_size_in_bytes() const { return _bvh_size_in_bytes; }

            virtual void clear() {
                _as_buffer_list.clear();
                optixDeviceContextDestroy(_optix_device_context);
            }

            NDSC std::string description() const {
                float size_in_M = (_bvh_size_in_bytes * 1.f) / (sqr(1024));
                return string_printf("bvh size is %f MB\n", size_in_M);
            }

            void build_bvh(const Buffer<const float3> &positions, const Buffer<const TriangleHandle> &triangles,
                           const Managed<MeshHandle> &meshes, const Managed<uint> &instance_list,
                           const Managed<Transform> &transform_list, const Managed<uint> &inst_to_transform);

        };
    }
}