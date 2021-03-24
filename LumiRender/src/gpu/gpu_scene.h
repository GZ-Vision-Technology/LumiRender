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

namespace luminous {
    inline namespace gpu {
        class GPUScene : public Scene {
        private:
            Buffer<uint> _instance_to_mesh_idx{nullptr};
            Buffer<uint> _instance_to_transform_idx{nullptr};
            Buffer<float4x4> _transforms{nullptr};

            Buffer<MeshHandle> _meshes{nullptr};
            Buffer<float3> _positions{nullptr};
            Buffer<float3> _normals{nullptr};
            Buffer<float2> _tex_coords{nullptr};
            Buffer<TriangleHandle> _triangles{nullptr};

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