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

            NDSC_XPU bool has_material() const {
                return material_idx != index_t(-1);
            }

            NDSC_XPU bool has_distribute() const {
                return distribute_idx != index_t(-1);
            }

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

        struct UVN {
            float3 dp_du;
            float3 dp_dv;
            float3 normal;

            XPU void set(float3 u, float3 v, float3 n) {
                dp_du = u;
                dp_dv = v;
                normal = n;
            }

            NDSC_XPU_INLINE bool valid() const {
                return !normal.is_zero();
            }
        };

        struct Interaction {
            float3 pos;
            float3 wo;
            float time;
            UVN g_uvn;

            NDSC_XPU_INLINE bool is_on_surface() const {
                return g_uvn.valid();
            }

            XPU Interaction() = default;
        };



        class Material;

        class Light;

        struct SurfaceInteraction : public Interaction {
            float2 uv;
            UVN s_uvn;
            float PDF_pos = 0;
            const Light *light = nullptr;
            const Material *material = nullptr;
            float du_dx = 0, dv_dx = 0, du_dy = 0, dv_dy = 0;
        };

        struct RadiancePRD {
            RadiancePRD() = default;

            RadiancePRD(bool count_emitted, bool done, bool hit,
                        luminous::float3 radiance,
                        luminous::float3 throughput,
                        luminous::float3 emission)
                    : count_emitted(count_emitted),
                      done(done),
                      hit(hit),
                      radiance(radiance),
                      throughput(throughput),
                      emission(emission) {}

            luminous::SurfaceInteraction interaction;
            bool count_emitted{true};
            bool done{false};
            bool hit{false};
            luminous::float3 radiance;
            luminous::float3 throughput;
            luminous::float3 emission;
        };

        struct TextureEvalContext {
            XPU TextureEvalContext() = default;

            XPU TextureEvalContext(const SurfaceInteraction &si)
                    : p(si.pos),
                      uv(si.uv),
                      du_dx(si.du_dx),
                      du_dy(si.du_dy),
                      dv_dx(si.dv_dx),
                      dv_dy(si.dv_dy) {}

            float3 p;
            float2 uv;
            float du_dx = 0, du_dy = 0, dv_dx = 0, dv_dy = 0;
        };

        struct MaterialEvalContext : public TextureEvalContext {
            XPU MaterialEvalContext() = default;

            XPU MaterialEvalContext(const SurfaceInteraction &si)
                    : TextureEvalContext(si),
                      wo(si.wo),
                      ng(si.g_uvn.normal),
                      ns(si.s_uvn.normal),
                      dp_dus(si.s_uvn.dp_du) {}

            float3 wo;
            float3 ng, ns;
            float3 dp_dus;
        };
    }
}