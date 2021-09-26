//
// Created by Zero on 2021/4/7.
//

#include "area_light.h"
#include "base_libs/sampling/distribution.h"
#include "render/scene/scene_data.h"

namespace luminous {
    inline namespace render {

        AreaLight::AreaLight(uint inst_idx, float3 L, float area, bool two_sided)
                : LightBase(LightType::Area),
                  _inst_idx(inst_idx),
                  _L(L),
                  _inv_area(1 / area),
                  _two_sided(two_sided) {
            volatile int a = 0;
//            printf("alignof area_light:%llu, sizeof:%llu\n", alignof(AreaLight), sizeof(AreaLight));
        }

        SurfaceInteraction AreaLight::sample(LightLiSample *lls, float2 u, const SceneData *scene_data) const {
            SurfaceInteraction ret;
            if (padded == 0.f)
                printf("padded %f \n", padded);
//            printf("----------alignof area_light:%llu, sizeof:%llu\n", alignof(AreaLight), sizeof(AreaLight));
            auto mesh = scene_data->get_mesh(_inst_idx);
            const Distribution1D &distrib = scene_data->distributions[mesh.distribute_idx];
            float PMF = 0;
            size_t triangle_id = distrib.sample_discrete(u.x, &PMF, &u.x);
            float2 uv = square_to_triangle(u);
            ret = scene_data->compute_surface_interaction(_inst_idx, triangle_id, uv);
            ret.PDF_pos = PMF / ret.prim_area;
            return ret;
        }

        LightLiSample AreaLight::Li(LightLiSample lls, const SceneData *data) const {
            float3 wi = lls.p_light.pos - lls.p_ref.pos;
            lls.wi = normalize(wi);
            lls.L = L(lls.p_light, -lls.wi);
            lls.PDF_dir = PDF_Li(lls.p_ref, lls.p_light, wi, data);
            return lls;
        }

        float AreaLight::PDF_Li(const Interaction &p_ref, const SurfaceInteraction &p_light,
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
            return (_two_sided ? _2Pi : Pi) * _L * (1.f / _inv_area);
        }

        Spectrum AreaLight::L(const SurfaceInteraction &p_light, float3 w) const {
            return (_two_sided || dot(w, p_light.g_uvn.normal) > 0) ? _L : make_float3(0.f);
        }

        void AreaLight::print() const {
            printf("type:AreaLight,instance id is %u,L:(%f,%f,%f)\n",
                   _inst_idx, _L.x, _L.y, _L.z);
        }

        CPU_ONLY(AreaLight AreaLight::create(const LightConfig &config) {
            return AreaLight(config.instance_idx, config.emission, config.surface_area,
                             config.two_sided);
        })

    } //luminous::render
} // luminous::render