//
// Created by Zero on 2021/2/1.
//


#pragma once

#include "core/backend/buffer.h"
#include "core/concepts.h"
#include "graphics/math/common.h"
#include "render/sensors/sensor.h"
#include "framework/optix_accel.h"

namespace luminous {
    inline namespace gpu {
        class GPUScene : public Noncopyable {
        private:
            Buffer<uint> _instance_to_mesh_idx{nullptr};
            Buffer<uint> _instance_to_transform_idx{nullptr};
            Buffer<float4x4> _transforms{nullptr};

            Buffer<MeshHandle> _meshes{nullptr};
            Buffer<float3> _positions{nullptr};
            Buffer<float3> _normals{nullptr};
            Buffer<float2> _tex_coords{nullptr};
            Buffer<TriangleHandle> _triangles{nullptr};

            vector<uint> _cpu_instance_to_mesh_idx{};
            vector<uint> _cpu_instance_to_transform_idx{};
            vector<float4x4> _cpu_transforms{};

            vector<MeshHandle> _cpu_meshes{};
            vector<float3> _cpu_positions{};
            vector<float3> _cpu_normals{};
            vector<float2> _cpu_tex_coords{};
            vector<TriangleHandle> _cpu_triangles{};

            SP<Device> _device;
            UP<OptixAccel> _optix_accel;

        public:
            GPUScene(const SP<Device> &device);

            void convert_geometry_data(const SP<SceneGraph> &scene_graph);

            void launch();

            void update_camera(const SensorHandle *camera);

            void build_accel();
        };
    }
}