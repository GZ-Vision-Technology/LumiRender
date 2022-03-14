//
// Created by Zero on 2021/4/7.
//

#include "area_light.h"
#include "render/scene/scene_data.h"
#include "core/refl/factory.h"

namespace luminous {
    inline namespace render {

        AreaLight::AreaLight(uint inst_idx, float3 L, bool two_sided)
                : LightBase(LightType::Area),
                  _inst_idx(inst_idx),
                  L(L),
                  _two_sided(two_sided) {}

        LightEvalContext AreaLight::sample(LightLiSample *lls, float2 u, const SceneData *scene_data) const {
            auto mesh = scene_data->get_mesh(_inst_idx);
            const Distribution1D &distrib = scene_data->distributions[mesh.distribute_idx];
            float PMF = 0;
            size_t triangle_id = distrib.sample_discrete(u.x, &PMF, &u.x);
            float2 bary = square_to_triangle(u);
            LightEvalContext lec;
            lec = scene_data->compute_light_eval_context(_inst_idx, triangle_id, bary);
            float3 wi_un = lec.pos - lls->lsc.pos;
            float PDF_dir = PDF_Li(lls->lsc, lec, wi_un, scene_data);
            lls->set_sample_result(PDF_dir, lec, normalize(wi_un));
            return lec;
        }

        LightLiSample AreaLight::Li(LightLiSample lls, const SceneData *data) const {
            lls.set_Li(radiance(lls.lec, -lls.wi, data));
            return lls;
        }

        float AreaLight::PDF_Li(const LightSampleContext &p_ref, const LightEvalContext &p_light,
                                float3 place_holder, const SceneData *data) const {
            float PDF = luminous::PDF_dir(p_light.PDF_pos, p_light.ng, p_ref.pos - p_light.pos);
            return select(is_inf(PDF), 0, PDF);
        }

        Spectrum AreaLight::power() const {
            // todo
            return (_two_sided ? _2Pi : Pi) * L/* *area */;
        }

        Spectrum AreaLight::radiance(const LightEvalContext &lec, float3 w,
                                     const SceneData *scene_data) const {
            return radiance(lec.uv, lec.ng, w, scene_data);
        }

        Spectrum AreaLight::radiance(float2 uv, float3 ng, float3 w, const SceneData *scene_data) const {
            return select(_two_sided || dot(w, ng) > 0, L, Spectrum{0.f});
        }

        void AreaLight::print() const {
            printf("type:AreaLight,instance id is %u,L:(%f,%f,%f)\n",
                   _inst_idx, L.x, L.y, L.z);
        }


    } //luminous::render
} // luminous::render