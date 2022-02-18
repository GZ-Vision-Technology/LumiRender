//
// Created by Zero on 2021/3/17.
//


#pragma once

#include "base_libs/math/common.h"
#include "core/backend/buffer_view.h"
#include "base_libs/geometry/util.h"
#include "base_libs/optics/rgb.h"
#include "base_libs/lstd/lstd.h"
#include "render/scattering/bsdf_wrapper.h"

namespace luminous {
    inline namespace render {

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

            ND_XPU_INLINE bool has_material() const {
                return material_idx != invalid_uint32;
            }

            ND_XPU_INLINE bool has_distribute() const {
                return distribute_idx != invalid_uint32;
            }

            ND_XPU_INLINE bool has_emission() const {
                return light_idx != invalid_uint32;
            }

            LM_XPU void print() const {
                printf("vert offset:%u, tri offset:%u, vert num:%u, tri num:%u, distribute idx: %u, mat idx %u\n",
                       vertex_offset,
                       triangle_offset,
                       vertex_count,
                       triangle_count,
                       distribute_idx,
                       material_idx);
            }
        };

        struct UVN : Frame {
            LM_XPU UVN() = default;

            LM_XPU void set_frame(Frame frame) {
                x = frame.x;
                y = frame.y;
                z = frame.z;
                CHECK_UNIT_VEC(x)
                CHECK_UNIT_VEC(y)
                CHECK_UNIT_VEC(z)
            }

            ND_XPU_INLINE float3 dp_du() const {
                return x;
            }

            ND_XPU_INLINE float3 dp_dv() const {
                return y;
            }

            ND_XPU_INLINE float3 normal() const {
                return z;
            }

            ND_XPU_INLINE bool valid() const {
                return nonzero(normal());
            }
        };

        struct Interaction {
            float3 pos;
            float3 wo;
            float time{};
            UVN g_uvn;

            LM_XPU Interaction() = default;

            LM_XPU explicit Interaction(float3 pos) : pos(pos) {}

            ND_XPU_INLINE bool is_on_surface() const {
                return g_uvn.valid();
            }

            ND_XPU_INLINE Ray spawn_ray(float3 dir) const {
                return Ray::spawn_ray(pos, g_uvn.normal(), dir);
            }

            ND_XPU_INLINE Ray spawn_ray_to(float3 p) const {
                return Ray::spawn_ray_to(pos, g_uvn.normal(), p);
            }

            ND_XPU_INLINE Ray spawn_ray_to(const Interaction &it) const {
                return Ray::spawn_ray_to(pos, g_uvn.normal(), it.pos, it.g_uvn.normal());
            }
        };


        class Material;

        class Light;

        struct SceneData;

        struct SurfaceInteraction : public Interaction {
            float2 uv;
            mutable UVN s_uvn;
            float PDF_pos{-1.f};
            float prim_area{0.f};
            const Light *light{nullptr};
            const Material *material{nullptr};
            float du_dx{0}, dv_dx{0}, du_dy{0}, dv_dy{0};

            LM_XPU SurfaceInteraction() = default;

            LM_XPU explicit SurfaceInteraction(float3 pos) : Interaction(pos) {}

            ND_XPU_INLINE bool has_emission() const {
                return light != nullptr;
            }

            ND_XPU_INLINE bool has_material() const {
                return material != nullptr;
            }

            LM_XPU void update_PDF_pos(float PMF) {
                PDF_pos = PMF / prim_area;
            }

            LM_ND_XPU Spectrum Le(float3 w, const SceneData *scene_data) const;

            LM_ND_XPU BSDFWrapper compute_BSDF(const SceneData *scene_data) const;
        };

        struct SurfacePoint {
            float3 pos;
            float3 ng;

            LM_XPU SurfacePoint() = default;

            LM_XPU SurfacePoint(float3 p, float3 n)
                    : pos(p), ng(n) {}

            LM_XPU explicit SurfacePoint(const Interaction &it)
                    : pos(it.pos), ng(it.g_uvn.normal()) {}

            LM_XPU explicit SurfacePoint(const SurfaceInteraction &it)
                    : pos(it.pos), ng(it.g_uvn.normal()) {}

            ND_XPU_INLINE Ray spawn_ray(float3 dir) const {
                return Ray::spawn_ray(pos, ng, dir);
            }

            ND_XPU_INLINE Ray spawn_ray_to(float3 p) const {
                return Ray::spawn_ray_to(pos, ng, p);
            }

            ND_XPU_INLINE Ray spawn_ray_to(const SurfacePoint &lsc) const {
                return Ray::spawn_ray_to(pos, ng, lsc.pos, lsc.ng);
            }
        };

        struct GeometrySurfacePoint : public SurfacePoint {
        public:
            float2 uv{};

            LM_XPU GeometrySurfacePoint() = default;

            LM_XPU explicit GeometrySurfacePoint(const Interaction &it, float2 uv)
                    : SurfacePoint(it), uv(uv) {}

            LM_XPU GeometrySurfacePoint(float3 p, float3 ng, float2 uv)
                    : SurfacePoint{p, ng}, uv(uv) {}
        };

        /**
         * A point on light
         * used to eval light PDF or lighting to LightSampleContext
         */
        struct LightEvalContext : public GeometrySurfacePoint {
        public:
            float PDF_pos{};
        public:
            LM_XPU LightEvalContext() = default;

            LM_XPU LightEvalContext(GeometrySurfacePoint gsp, float PDF_pos)
                    : GeometrySurfacePoint(gsp), PDF_pos(PDF_pos) {}

            LM_XPU LightEvalContext(float3 p, float3 ng, float2 uv, float PDF_pos)
                    : GeometrySurfacePoint{p, ng, uv}, PDF_pos(PDF_pos) {}

            LM_XPU explicit LightEvalContext(const SurfaceInteraction &si)
                    : GeometrySurfacePoint{si, si.uv}, PDF_pos(si.PDF_pos) {}
        };

        struct HitContext {
            HitInfo hit_info{};
            const SceneData *data{nullptr};

            HitContext() = default;

            LM_XPU HitContext(HitInfo hit_info, const SceneData *data)
                    : hit_info(hit_info), data(data) {}

            LM_XPU explicit HitContext(const SceneData *data)
                    : data(data) {}

            ND_XPU_INLINE bool is_hit() const {
                return hit_info.is_hit();
            }

            LM_ND_XPU bool has_emission() const;

            LM_ND_XPU bool has_material() const;

            /**
             * compute geometry data world position and
             * @return position in world space, geometry normal in world space
             */
            LM_ND_XPU GeometrySurfacePoint geometry_surface_point() const;

            LM_ND_XPU SurfacePoint surface_point() const;

            LM_ND_XPU float compute_prim_PMF() const;

            LM_ND_XPU const Light *light() const;

            LM_ND_XPU LightEvalContext compute_light_eval_context() const;

            LM_ND_XPU const Material *material() const;

            LM_ND_XPU SurfaceInteraction compute_surface_interaction(float3 wo) const;

            LM_ND_XPU SurfaceInteraction compute_surface_interaction(Ray ray) const;

            ND_XPU_INLINE const SceneData *scene_data() const { return data; }
        };

        struct NEEData {
            Spectrum bsdf_val{0.f};
            Spectrum albedo{};
            float bsdf_PDF{-1.f};
            float3 wi{0.f};
            BxDFFlags bxdf_flags{Unset};
            float eta{1.f};
            SurfaceInteraction next_si;
            bool found_intersection{false};
            bool debug = false;
        };

        struct TextureEvalContext {
            LM_XPU TextureEvalContext() = default;

            LM_XPU explicit TextureEvalContext(float2 uv)
                    : uv(uv) {}

            LM_XPU TextureEvalContext(const SurfaceInteraction &si)
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
            LM_XPU MaterialEvalContext() = default;

            LM_XPU MaterialEvalContext(const SurfaceInteraction &si)
                    : TextureEvalContext(si),
                      wo(si.wo),
                      ng(si.g_uvn.normal()),
                      ns(si.s_uvn.normal()),
                      dp_dus(si.s_uvn.dp_du()) {}

            float3 wo;
            float3 ng, ns;
            float3 dp_dus;
        };
    }
}