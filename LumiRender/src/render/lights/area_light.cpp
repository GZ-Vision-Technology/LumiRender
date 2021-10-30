//
// Created by Zero on 2021/4/7.
//

#include "area_light.h"
#include "render/scene/scene_data.h"
#include "core/refl/factory.h"

namespace luminous {
    inline namespace render {

        AreaLight::AreaLight(uint inst_idx, float3 L, float area, bool two_sided)
                : BaseBinder<LightBase>(LightType::Area),
                  _inst_idx(inst_idx),
                  L(L),
                  _inv_area(1 / area),
                  _two_sided(two_sided) {}

        SurfaceInteraction AreaLight::sample(LightLiSample *lls, float2 u, const SceneData *scene_data) const {
            SurfaceInteraction ret;
            auto mesh = scene_data->get_mesh(_inst_idx);
            const Distribution1D &distrib = scene_data->distributions[mesh.distribute_idx];
            float PMF = 0;
            size_t triangle_id = distrib.sample_discrete(u.x, &PMF, &u.x);
            float2 uv = square_to_triangle(u);
            ret = scene_data->compute_surface_interaction(_inst_idx, triangle_id, uv);
            ret.PDF_pos = PMF * ret.PDF_pos;
            return ret;
        }

        LightLiSample AreaLight::Li(LightLiSample lls, const SceneData *data) const {
            float3 wi = lls.p_light.pos - lls.ctx.pos;
            lls.wi = normalize(wi);
            lls.L = radiance(lls.p_light, -lls.wi, data);
            lls.PDF_dir = PDF_Li(lls.ctx, lls.p_light, wi, data);
            return lls;
        }

        float AreaLight::PDF_Li(const LightSampleContext &p_ref, const SurfaceInteraction &p_light,
                                float3 wi, const SceneData *data) const {
            float3 wo = p_ref.pos - p_light.pos;
            float PDF = luminous::PDF_dir(p_light.PDF_pos, p_light.g_uvn.normal, wo);
            if (is_inf(PDF)) {
                return 0;
            }
            return PDF;
        }

        float AreaLight::inv_area() const {
            return _inv_area;
        }

        Spectrum AreaLight::power() const {
            return (_two_sided ? _2Pi : Pi) * L * (1.f / _inv_area);
        }

        Spectrum AreaLight::radiance(const SurfaceInteraction &p_light, float3 w,
                                     const SceneData *scene_data) const {
            return radiance(p_light.uv, p_light.g_uvn.normal, w, scene_data);
        }

        Spectrum AreaLight::radiance(float2 uv, float3 ng, float3 w, const SceneData *scene_data) const {
            return (_two_sided || dot(w, ng) > 0) ? L : make_float3(0.f);
        }

        void AreaLight::print() const {
            printf("type:AreaLight,instance id is %u,L:(%f,%f,%f)\n",
                   _inst_idx, L.x, L.y, L.z);
        }


    } //luminous::render
} // luminous::render