//
// Created by Zero on 2021/2/1.
//


#pragma once

#include "core/backend/buffer.h"
#include "core/concepts.h"
#include "graphics/math/common.h"
#include "render/sensors/sensor.h"
#include "framework/optix_accel.h"
#include "render/include/scene.h"
#include "core/backend/managed.h"

namespace luminous {
    inline namespace gpu {
        class GPUScene : public Scene {
        private:
            Buffer<uint> _instance_to_mesh_idx{nullptr};
            Buffer<uint> _instance_to_transform_idx{nullptr};
            Buffer<float4x4> transforms{nullptr};

            Buffer<MeshHandle> meshes{nullptr};
            Buffer<float3> positions{nullptr};
            Buffer<float3> normals{nullptr};
            Buffer<float2> tex_coords{nullptr};
            Buffer<TriangleHandle> triangles{nullptr};


            Managed<uint*> _inst_to_mesh_idx;
            Managed<uint*> _inst_to_transform_idx;
            Managed<float4x4*> _transforms;

            Managed<MeshHandle*> _meshes;
            Managed<float3*> _positions;
            Managed<float3*> _normals;
            Managed<float2*> _tex_coords;
            Managed<TriangleHandle*> _triangles;


            SP<Device> _device;
            UP<OptixAccel> _optix_accel;

        public:
            GPUScene(const SP<Device> &device);

            void init(const SP<SceneGraph> &scene_graph) override;

            void create_device_memory();

            void launch();

            void build_accel();
        };
    }
}