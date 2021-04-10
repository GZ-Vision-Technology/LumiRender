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

namespace luminous {
    inline namespace render {
        using namespace std;
        class Scene : public Noncopyable {
        protected:
            vector<uint> _cpu_inst_to_mesh_idx{};
            vector<uint> _cpu_inst_to_transform_idx{};
            vector<float4x4> _cpu_transforms{};

            vector<MeshHandle> _cpu_meshes{};
            vector<float3> _cpu_positions{};
            vector<float3> _cpu_normals{};
            vector<float2> _cpu_tex_coords{};
            vector<TriangleHandle> _cpu_triangles{};

            vector<Light> _cpu_lights{};

            size_t _inst_vertices_num{0};
            size_t _inst_triangle_num{0};

        public:

            void shrink_to_fit();

            NDSC virtual std::string description() const;

            virtual size_t size_in_bytes() const;

            void clear();

            virtual void init(const SP<SceneGraph> &scene_graph) = 0;

            virtual void init_accel() = 0;

            void convert_data(const SP<SceneGraph> &scene_graph);

            void load_lights(const vector<LightConfig> &lc);
        };
    }
}