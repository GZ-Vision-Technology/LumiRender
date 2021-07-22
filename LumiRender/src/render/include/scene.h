//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "core/concepts.h"
#include "graphics/math/common.h"
#include "shape.h"
#include "scene_graph.h"
#include "render/lights/light.h"
#include "graphics/sampling/common.h"
#include "render/include/emission_distribution.h"
#include "render/light_samplers/light_sampler.h"
#include "render/textures/texture.h"
#include "gpu/gpu_include.h"
#include "render/materials/material.h"

namespace luminous {
    inline namespace render {

        template<typename T>
        void append(std::vector<T> &v1, const std::vector<T> &v2) {
            v1.insert(v1.cend(), v2.cbegin(), v2.cend());
        }

        class Scene : public Noncopyable {
        protected:
            Context *_context{nullptr};
            float3 _bg_color = make_float3(0.f);
            size_t _inst_vertices_num{0};
            size_t _inst_triangle_num{0};
            size_t _texture_size_in_byte{0};
            size_t _texture_num{0};

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
            EmissionDistribution _emission_distrib;
            Managed<LightSampler> _light_sampler;

            // texture data
            Managed<Texture> _textures;

            // material data, the last element is light material, black diffuse
            Managed<Material> _materials;

            // texture manager, manage the texture on video memory
            vector <DeviceTexture> _texture_mgr;

            // prepare for texture out of core render
            vector<TextureConfig> _tex_configs;

        public:
            Scene(Context *context) : _context(context) {}

            void shrink_to_fit();

            NDSC virtual std::string description() const;

            virtual size_t size_in_bytes() const;

            virtual void clear();

            void init_materials(const SP<SceneGraph> &scene_graph);

            virtual void init(const SP<SceneGraph> &scene_graph) = 0;

            virtual void init_accel() = 0;

            void append_light_material(vector <MaterialConfig> &material_configs);

            void convert_data(const SP<SceneGraph> &scene_graph);

            void relevance_material_and_texture(vector<MaterialConfig> &material_configs);

            void load_lights(const vector <LightConfig> &lc, const LightSamplerConfig &lsc);

            void preprocess_meshes();
        };
    }
}