//
// Created by Zero on 2021/4/7.
//

#include "point_light.h"

namespace luminous {
    inline namespace render {

        Interaction PointLight::sample(float u, const HitGroupData *hit_group_data) const {
            Interaction ret;
            ret.pos = _pos;
            return ret;
        }

        float PointLight::PDF_Li(const Interaction &ref_p, float3 wi) const {
            return 0;
        }

        float3 PointLight::power() const {
            return 4 * Pi * _intensity;
        }

        std::string PointLight::to_string() const {
            return string_printf("light Base : %s, intensity : %s",
                                 _to_string().c_str(),
                                 _intensity.to_string().c_str());
        }

        LightLiSample PointLight::Li(LightLiSample lls) const {
            float3 wi = lls.p_light.pos - lls.p_ref.pos;
            lls.L = _intensity / length_squared(wi);
            lls.PDF_dir = 0;
            lls.wi = normalize(wi);
            return lls;
        }

        PointLight PointLight::create(const LightConfig &config) {
            return PointLight(config.position, config.intensity);
        }
    } // luminous::render
} // luminous