//
// Created by Zero on 2021/3/17.
//


#pragma once

#include "base_libs/math/common.h"
#include "core/backend/buffer_view.h"
#include "base_libs/geometry/util.h"
#include "base_libs/optics/rgb.h"
#include "base_libs/lstd/lstd.h"
#include "render/bxdfs/bsdf.h"

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
                       index_t distribute_idx = -1,
                       index_t light_idx = -1)
                    : vertex_offset(vert_ofs),
                      triangle_offset(tri_ofs),
                      vertex_count(vert_count),
                      triangle_count(tri_count),
                      material_idx(material_idx),
                      distribute_idx(distribute_idx),
                      light_idx(light_idx) {}

            index_t vertex_offset;
            index_t triangle_offset;
            index_t vertex_count;
            index_t triangle_count;
            index_t distribute_idx;
            index_t material_idx;
            index_t light_idx;

            NDSC_XPU_INLINE bool has_material() const {
                return material_idx != invalid_uint32;
            }

            NDSC_XPU_INLINE bool has_distribute() const {
                return distribute_idx != invalid_uint32;
            }

            NDSC_XPU_INLINE bool has_light() const {
                return light_idx != invalid_uint32;
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
                return nonzero(normal);
            }
        };

        struct Interaction {
            float3 pos;
            float3 wo;
            float time{};
            UVN g_uvn;

            XPU Interaction() = default;

            XPU explicit Interaction(float3 pos) : pos(pos) {}

            NDSC_XPU_INLINE bool is_on_surface() const {
                return g_uvn.valid();
            }

            NDSC_XPU_INLINE Ray spawn_ray(float3 dir) const {
                return Ray::spawn_ray(pos, g_uvn.normal, dir);
            }

            NDSC_XPU_INLINE Ray spawn_ray_to(float3 p) const {
                return Ray::spawn_ray_to(pos, g_uvn.normal, p);
            }

            NDSC_XPU_INLINE Ray spawn_ray_to(const Interaction &it) const {
                return Ray::spawn_ray_to(pos, g_uvn.normal, it.pos, it.g_uvn.normal);
            }
        };


        class Material;

        class Light;

        class SceneData;

        class MissData;

        struct SurfaceInteraction : public Interaction {
            float2 uv;
            UVN s_uvn;
            float PDF_pos = 0;
            float prim_area = 0;
            const Light *light = nullptr;
            lstd::optional<BSDF> op_bsdf{};
            const Material *material = nullptr;
            float du_dx = 0, dv_dx = 0, du_dy = 0, dv_dy = 0;

            XPU SurfaceInteraction() = default;

            XPU explicit SurfaceInteraction(float3 pos) : Interaction(pos) {}

            NDSC_XPU_INLINE bool has_emission() const {
                return light != nullptr;
            }

            NDSC_XPU_INLINE bool has_material() const {
                return material != nullptr;
            }

            NDSC_XPU Spectrum Le(float3 w) const;

            NDSC_XPU lstd::optional<BSDF> get_BSDF(const SceneData *scene_data) const;

            XPU_INLINE void init_BSDF(const SceneData *scene_data) {
                op_bsdf = get_BSDF(scene_data);
            }
        };

        class Sampler;

        struct PerRayData {
            ClosestHit closest_hit{};
            const SceneData *data{nullptr};

            PerRayData() = default;

            explicit PerRayData(const SceneData *data)
                    : data(data) {}

            NDSC_XPU_INLINE bool is_hit() const {
                return closest_hit.is_hit();
            }

            NDSC_XPU SurfaceInteraction compute_surface_interaction(Ray ray) const;

            NDSC_XPU_INLINE const SceneData *scene_data() const { return data; }
        };

        struct NEEData {
            Spectrum bsdf_val{0.f};
            float bsdf_PDF{-1.f};
            float3 wi{0.f};
            SurfaceInteraction next_si;
            bool found_intersection{false};
            bool debug = false;
        };

        struct TextureEvalContext {
            XPU TextureEvalContext() = default;

            XPU explicit TextureEvalContext(float2 uv)
                    : uv(uv) {}

            XPU TextureEvalContext(const SurfaceInteraction &si)
                    : p(si.pos),
                      uv(si.uv),
                      du_dx(si.du_dx),
                      du_dy(si.du_dy),
                      dv_dx(si.dv_dx),
                      dv_dy(si.dv_dy) {}

            float3 p{};
            float2 uv{};
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