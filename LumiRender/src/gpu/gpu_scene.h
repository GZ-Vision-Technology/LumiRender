//
// Created by Zero on 2021/2/1.
//


#pragma once

#include "core/backend/buffer.h"
#include "core/concepts.h"
#include "graphics/math/common.h"
#include "render/sensors/camera_base.h"
#include "framework/optix_accel.h"
#include "render/include/scene.h"
#include "core/backend/managed.h"

namespace luminous {
    inline namespace gpu {

        struct EmissionDistributeData {
            struct DistributeHandle {
                DistributeHandle() = default;

                DistributeHandle(uint func_offset,
                                 uint func_count,
                                 uint CDF_offset,
                                 uint CDF_count,
                                 uint integral)
                        : func_offset(func_offset),
                          func_count(func_count),
                          CDF_offset(CDF_offset),
                          CDF_count(CDF_count),
                          integral(integral) {}

                uint func_offset;
                uint func_count;
                uint CDF_offset;
                uint CDF_count;
                uint integral;
            };

            Managed<float> func;
            Managed<float> CDF;
            vector<DistributeHandle> handles;
        };


        class GPUScene : public Scene {
        private:
            // instance data
            Managed<uint> _inst_to_mesh_idx;
            Managed<uint> _inst_to_transform_idx;
            Managed<float4x4> _transforms;

            // mesh data
            Managed<MeshHandle> _meshes;
            Managed<float3> _positions;
            Managed<float3> _normals;
            Managed<float2> _tex_coords;
            Managed<TriangleHandle> _triangles;

            // light data
            Managed<Light> _lights;
            Managed<Distribute1D> _emission_distributes;
            EmissionDistributeData _emission_distribute_data;

            float3 _bg_color = make_float3(0.f);

            SP<Device> _device;
            UP<OptixAccel> _optix_accel;

            friend class OptixAccel;

        public:
            GPUScene(const SP<Device> &device);

            void init(const SP<SceneGraph> &scene_graph) override;

            void init_accel() override;

            void build_emission_distribute() override;

            size_t size_in_bytes() const override;

            void create_device_memory();

            inline void clear_host() {
                Scene::clear();
            }

            void synchronize_to_gpu();

            template<typename... Args>
            void launch(Args &&...args) {
                _optix_accel->launch(std::forward<Args>(args)...);
            }

            void build_accel();

            NDSC std::string description() const override;
        };
    }
}