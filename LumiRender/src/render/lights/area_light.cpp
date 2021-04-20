//
// Created by Zero on 2021/4/7.
//

#include "area_light.h"

namespace luminous {
    inline namespace render {

        Interaction AreaLight::sample(float u, const HitGroupData *hit_group_data) const {
            Interaction ret;
            // todo
            return ret;
        }

        LightLiSample AreaLight::Li(LightLiSample lls) const {
            lls.wi = normalize(lls.p_light.pos - lls.p_ref.pos);
            lls.L = L(lls.p_light, -lls.wi);
            float PDF_pos = _inv_area;
            float cos_theta = abs_dot(lls.p_light.ng, normalize(-lls.wi));
            float PDF_dir = PDF_pos * length_squared(lls.wi) / cos_theta;
            if (is_inf(PDF_dir)) {
                PDF_dir = 0;
            }
            lls.PDF_dir = PDF_dir;
            return lls;
        }

        /**
         * p(dir) = p(pos) * r^2 / cos¦È
         * @param p_ref
         * @param p_light
         * @return
         */
        float AreaLight::PDF_Li(const Interaction &p_ref, const Interaction &p_light) const {
            float PDF_pos = _inv_area;
            float3 wi = p_ref.pos - p_light.pos;
            float cos_theta = abs_dot(p_light.ng, normalize(wi));
            float PDF_dir = PDF_pos * length_squared(wi) / cos_theta;
            if (is_inf(PDF_dir)) {
                return 0;
            }
            return PDF_dir;
        }

        float3 AreaLight::power() const {
            return (_two_sided ? _2Pi : Pi) * _L / _inv_area;
        }

        float3 AreaLight::L(const Interaction &p_light, float3 w) const {
            return (_two_sided || dot(w, p_light.ng)) ? _L : make_float3(0.f);
        }

        std::string AreaLight::to_string() const {
            LUMINOUS_TO_STRING("light Base : %s,name:%s, L : %s",
                                 _to_string().c_str(),
                                 name().c_str(),
                                 _L.to_string().c_str());
        }

        AreaLight AreaLight::create(const LightConfig &config) {
            return AreaLight(config.instance_idx, config.emission);
        }
    } //luminous::render
} // luminous::render