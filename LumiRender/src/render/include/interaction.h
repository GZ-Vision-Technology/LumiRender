//
// Created by Zero on 2021/3/17.
//


#pragma once

#include "graphics/math/common.h"

namespace luminous {
    inline namespace render {
        struct Interaction {
            float3 pos;
            float3 ng;
            float3 ns;
            float2 uv;
            float3 wo;
            float time;
        };

        struct TriangleHandle {
            uint i;
            uint j;
            uint k;
            XPU void print() const {
                printf("i:%u, j:%u, k:%u \n", i, j, k);
            }
        };

        struct MeshHandle {
            MeshHandle() = default;

            MeshHandle(uint vert_ofs, uint tri_ofs,
                       uint vert_count, uint tri_count,
                       int distribute_idx = -1)
                    : vertex_offset(vert_ofs),
                      triangle_offset(tri_ofs),
                      vertex_count(vert_count),
                      triangle_count(tri_count),
                      distribute_idx(distribute_idx) {}

            uint vertex_offset;
            uint triangle_offset;
            uint vertex_count;
            uint triangle_count;
            int distribute_idx;

            XPU void print() const {
                printf("vert offset:%u, tri offset:%u, vert num:%u, tri num:%u, distribute idx: %d\n",
                       vertex_offset,
                       triangle_offset,
                       vertex_count,
                       triangle_count,
                       distribute_idx);
            }
        };

        struct HitGroupData {
            // instance data
            uint *inst_to_mesh_idx;
            uint *inst_to_transform_idx;
            float4x4 *transforms;
            // mesh data
            MeshHandle *meshes;
            float3 *positions;
            float3 *normals;
            float2 *tex_coords;
            TriangleHandle *triangles;
        };
    }
}