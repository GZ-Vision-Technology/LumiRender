//
// Created by Zero on 2021/1/30.
//


#pragma once

#include "graphics/math/common.h"
#include <vector>
#include "core/concepts.h"

namespace luminous {
    inline namespace render {
        using std::vector;

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
            float _surface_area;
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
    }
}