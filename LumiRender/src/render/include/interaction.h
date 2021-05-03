//
// Created by Zero on 2021/3/17.
//


#pragma once

#include "graphics/math/common.h"
#include "core/backend/buffer_view.h"

namespace luminous {
    inline namespace render {

        struct TriangleHandle {
            index_t i;
            index_t j;
            index_t k;
            XPU void print() const {
                printf("i:%u, j:%u, k:%u \n", i, j, k);
            }
        };

        struct MeshHandle {
            MeshHandle() = default;

            MeshHandle(index_t vert_ofs, index_t tri_ofs,
                       index_t vert_count, index_t tri_count,
                       index_t material_idx = -1,
                       index_t distribute_idx = -1)
                    : vertex_offset(vert_ofs),
                      triangle_offset(tri_ofs),
                      vertex_count(vert_count),
                      triangle_count(tri_count),
                      material_idx(material_idx),
                      distribute_idx(distribute_idx) {}

            index_t vertex_offset;
            index_t triangle_offset;
            index_t vertex_count;
            index_t triangle_count;
            index_t distribute_idx;
            index_t material_idx;

            XPU void print() const {
                printf("vert offset:%u, tri offset:%u, vert num:%u, tri num:%u, distribute idx: %u, mat idx %u\n",
                       vertex_offset,
                       triangle_offset,
                       vertex_count,
                       triangle_count,
                       distribute_idx,
                       material_idx);
            }
        };

        struct Interaction {
            float3 pos;
            float3 wo;
            float time;

            Interaction() = default;
        };

        struct UVN {
            float3 dp_du;
            float3 dp_dv;
            float3 normal;

            XPU void set(float3 u, float3 v, float3 n) {
                dp_du = u;
                dp_dv = v;
                normal = n;
            }
        };

        struct SurfaceInteraction : public Interaction {
            float2 uv;
            UVN s_uvn;
            UVN g_uvn;
            float3 dp_dx;
            float3 dp_dy;
            float du_dx = 0, dv_dx = 0, du_dy = 0, dv_dy = 0;
        };
    }
}