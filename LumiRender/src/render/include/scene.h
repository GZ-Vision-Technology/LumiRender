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

namespace luminous {
    inline namespace render {
        using namespace std;
        class Scene : public Noncopyable {
        protected:
            size_t _inst_vertices_num{0};
            size_t _inst_triangle_num{0};

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
            vector<Distribution1DBuilder> _emission_distribution_builders;
            Managed<Light> _lights;
            EmissionDistribution _emission_distrib;

        public:

            void shrink_to_fit();

            NDSC virtual std::string description() const;

            virtual size_t size_in_bytes() const;

            void clear();

            virtual void init(const SP<SceneGraph> &scene_graph) = 0;

            virtual void init_accel() = 0;

            virtual void build_emission_distribute();

            void convert_data(const SP<SceneGraph> &scene_graph);

            void load_lights(const vector<LightConfig> &lc);

            void preprocess_meshes();
        };
    }
}