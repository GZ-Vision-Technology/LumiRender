//
// Created by Zero on 2021/1/30.
//


#pragma once

#include "graphics/geometry/common.h"
#include <vector>
#include "core/concepts.h"

namespace luminous {

    inline namespace render {
        using std::vector;
        using std::string;

        class Shape : public IObject {

        };

        struct TriangleHandle {
            uint i;
            uint j;
            uint k;
        };

        struct MeshHandle {
            uint vertex_offset;
            uint triangle_offset;
            uint vertex_count;
            uint triangle_count;
        };

        struct Mesh : public Shape {
            Mesh(vector<float3> P,
                 vector<float3> N,
                 vector<float2> uv,
                 vector<TriangleHandle> T) :
                    positions(std::move(P)),
                    normals(std::move(N)),
                    tex_coords(std::move(uv)),
                    triangles(std::move(T)) {}

            vector<float3> normals;
            vector<float3> positions;
            vector<float2> tex_coords;
            vector<TriangleHandle> triangles;
        };

        struct Model {
            Model(const std::filesystem::path &fn, uint subdiv_level = 0);

            vector<std::shared_ptr<const Mesh>> meshes;
            string key;
        };

        struct ModelInstance {
            ModelInstance(uint idx, const Transform &t, const char *n)
                    : model_idx(idx),
                      o2w(t),
                      name(n) {}

            const char *name;
            const uint model_idx;
            const Transform o2w;
        };
    }

}