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

        struct ModelHandle {
            uint vertex_offset;
            uint triangle_offset;
        };

        struct MeshHandle : public ModelHandle {
            uint vertex_count;
            uint triangle_count;
        };

        class Mesh : public Shape {
        private:
            vector<float3> _normals;
            vector<float3> _positions;
            vector<float2> _tex_coords;
            vector<TriangleHandle> _triangles;
        public:
            Mesh(vector<float3> P,
                 vector<float3> N,
                 vector<float2> uv,
                 vector<TriangleHandle> T) :
                    _positions(std::move(P)),
                    _normals(std::move(N)),
                    _tex_coords(std::move(uv)),
                    _triangles(std::move(T)) {}
        };

        class Model {
        private:
            vector<std::shared_ptr<const Mesh>> _meshes;
        public:
            string key;

            Model(const std::filesystem::path &fn, uint subdiv_level = 0);
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