//
// Created by Zero on 2021/4/7.
//

#include "area_light.h"
#include "render/include/creator.h"
#include "render/include/distribution.h"
#include "graphics/sampling/warp.h"

namespace luminous {
    inline namespace render {

        SurfaceInteraction AreaLight::sample(float2 u, const HitGroupData *hit_group_data) const {
            SurfaceInteraction ret;
            auto mesh = hit_group_data->get_mesh(_inst_idx);
            const Distribution1D &distrib = hit_group_data->emission_distributions[mesh.distribute_idx];
            float PMF = 0;
            size_t triangle_id = distrib.sample_discrete(u.x, &PMF, &u.x);
            float2 uv = square_to_triangle(u);
            ret = hit_group_data->compute_surface_interaction(_inst_idx, triangle_id, uv);
            ret.PDF_pos = ret.prim_area / 1.f * PMF;
            return ret;
        }

        LightLiSample AreaLight::Li(LightLiSample lls) const {
            lls.wi = normalize(lls.p_light.pos - lls.p_ref.pos);
            lls.L = L(lls.p_light, -lls.wi);
            float cos_theta = abs_dot(lls.p_light.g_uvn.normal, normalize(-lls.wi));
            float PDF_dir = lls.p_light.PDF_pos * length_squared(lls.wi) / cos_theta;
            if (is_inf(PDF_dir)) {
                PDF_dir = 0;
            }
            lls.PDF_dir = PDF_dir;
            return lls;
        }

        lstd::optional<LightLiSample> AreaLight::sample_Li(float2 u, LightLiSample lls,
                                                           uint64_t traversable_handle,
                                                           const HitGroupData *hit_group_data) const {
            lls.p_light = sample(u, hit_group_data);
            Ray ray = lls.p_ref.spawn_ray_to(lls.p_light);
            bool occluded = ray_occluded(traversable_handle, ray);
            if (occluded) {
                return {};
            }
            lls = Li(lls);
            return lls;
        }

        /**
         * p(dir) = p(pos) * r^2 / cos��
         * @param p_ref
         * @param p_light
         * @return
         */
        float AreaLight::PDF_dir(const Interaction &p_ref, const SurfaceInteraction &p_light) const {
            float3 wi = p_ref.pos - p_light.pos;
            float cos_theta = abs_dot(p_light.g_uvn.normal, normalize(wi));
            float PDF = p_light.PDF_pos * length_squared(wi) / cos_theta;
            if (is_inf(PDF)) {
                return 0;
            }
            return PDF;
        }

        Spectrum AreaLight::power() const {
            return (_two_sided ? _2Pi : Pi) * _L * _area;
        }

        Spectrum AreaLight::L(const SurfaceInteraction &p_light, float3 w) const {
            return (_two_sided || dot(w, p_light.g_uvn.normal)) ? _L : make_float3(0.f);
        }

        std::string AreaLight::to_string() const {
            LUMINOUS_TO_STRING("light Base : %s,name:%s, L : %s",
                               _to_string().c_str(),
                               type_name(this),
                               _L.to_string().c_str());
        }

        AreaLight AreaLight::create(const LightConfig &config) {
            return AreaLight(config.instance_idx, config.emission, config.surface_area);
        }


    } //luminous::render
} // luminous::render