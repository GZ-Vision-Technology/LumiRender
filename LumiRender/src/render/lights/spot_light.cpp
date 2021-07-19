//
// Created by Zero on 2021/4/8.
//


#include "spot_light.h"

namespace luminous {
    inline namespace render {


        Interaction SpotLight::sample(float u, const HitGroupData *hit_group_data) const {
            return Interaction();
        }

        LightLiSample SpotLight::Li(LightLiSample lls) const {
            return LightLiSample();
        }

        float SpotLight::PDF_Li(const Interaction &ref_p, const Interaction &p_light) const {
            return 0;
        }

        float3 SpotLight::power() const {
            return _intensity * 2.f * Pi * (1 - 0.5f * (_cos_theta_i + _cos_theta_o));
        }
    } // luminous::render
} // luminous