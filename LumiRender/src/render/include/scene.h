//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "core/concepts.h"
#include "base_libs/math/common.h"
#include "shape.h"
#include "scene_graph.h"
#include "render/lights/light.h"
#include "base_libs/sampling/common.h"
#include "render/light_samplers/light_sampler.h"
#include "render/textures/texture.h"
#include "distribution_mgr.h"
#include "render/materials/material.h"
#include "render/include/accelerator.h"

namespace luminous {

    inline namespace gpu {
        class MegakernelOptixAccel;
        class OptixAccel;
        class ShaderWrapper;
    }

    inline namespace render {

        template<typename T>
        void append(std::vector<T> &v1, const std::vector<T> &v2) {
            v1.insert(v1.cend(), v2.cbegin(), v2.cend());
        }

        /**
         * all memory data manage
         */
        class Scene : public Noncopyable {
            friend class gpu::MegakernelOptixAccel;
            friend class gpu::OptixAccel;
            friend class gpu::ShaderWrapper;
        protected:
            SP<Device> _device;
            Context *_context{nullptr};
            size_t _inst_vertices_num{0};
            size_t _inst_triangle_num{0};
            size_t _texture_size_in_byte{0};
            size_t _texture_num{0};
            int _infinite_light_num{0};

            Box3f _scene_box;

            // instance data
            Managed<uint> _inst_to_mesh_idx;
            Managed<uint> _inst_to_transform_idx;
            Managed<Transform> _transforms;

            // mesh data
            Managed<MeshHandle> _meshes;
            Managed<float3> _positions;
            Managed<float3> _normals;
            Managed<float2> _tex_coords;
            Managed<TriangleHandle> _triangles;

            // light data
            Managed<Light> _lights;
            Managed<LightSampler> _light_sampler;
            DistributionMgr _distribution_mgr;

            // texture data
            Managed<Texture> _textures;

            // material data, the last element is light material, black diffuse
            Managed<Material> _materials;

            // prepare for texture out of core render
            vector<TextureConfig> _tex_configs;

            vector<Image> _images;

            UP<Accelerator> _accelerator;

        public:
            Scene(const SP<Device> &device, Context *context)
                    : _device(device), _context(context) {}

            void shrink_to_fit();

            NDSC virtual std::string description() const;

            NDSC virtual size_t size_in_bytes() const;

            virtual void clear();

            NDSC uint64_t as_handle() const { return _accelerator->handle(); }

            template<typename TAccel>
            NDSC decltype(auto) accel() {
                return reinterpret_cast<TAccel*>(_accelerator.get());
            }

            void init_materials(const SP<SceneGraph> &scene_graph);

            virtual void init(const SP<SceneGraph> &scene_graph) = 0;

            virtual void create_device_memory() = 0;

            // todo add geometry accelerate structure
            virtual void init_accel() = 0;

            void append_light_material(vector<MaterialConfig> &material_configs);

            void convert_geometry_data(const SP<SceneGraph> &scene_graph);

            void init_lights(const SP<SceneGraph> &scene_graph);

            void preload_textures(const SP<SceneGraph> &scene_graph);

            void relevance_material_and_texture(vector<MaterialConfig> &material_configs);

            void relevance_light_and_texture(vector<LightConfig> &light_configs);

            void load_lights(const vector<LightConfig> &lc, const LightSamplerConfig &lsc);

            void preprocess_meshes();
        };
    }
}