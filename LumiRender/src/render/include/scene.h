//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "core/concepts.h"
#include "graphics/math/common.h"
#include "shape.h"
#include "scene_graph.h"

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


        public:

            size_t size_in_bytes() const;

            virtual void init(const SP<SceneGraph> &scene_graph) = 0;

            void convert_geometry_data(const SP<SceneGraph> &scene_graph);
        };
    }
}