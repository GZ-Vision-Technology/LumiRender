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
            // instance data
            Managed_old<uint*> _inst_to_mesh_idx;
            Managed_old<uint*> _inst_to_transform_idx;
            Managed_old<float4x4*> _transforms;
            // mesh data
            Managed_old<MeshHandle*> _meshes;
            Managed_old<float3*> _positions;
            Managed_old<float3*> _normals;
            Managed_old<float2*> _tex_coords;
            Managed_old<TriangleHandle*> _triangles;

            SP<Device> _device;
            UP<OptixAccel> _optix_accel;

        public:
            GPUScene(const SP<Device> &device);

            void init(const SP<SceneGraph> &scene_graph) override;

            void create_device_memory();

            void synchronize_to_gpu();

            template<typename... Args>
            void launch(Args &&...args) {
                _optix_accel->launch(std::forward<Args>(args)...);
            }

            void build_accel();
        };
    }
}