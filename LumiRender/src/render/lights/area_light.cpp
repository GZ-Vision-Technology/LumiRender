//
// Created by Zero on 2021/4/7.
//

#include "area_light.h"

namespace luminous {
    inline namespace render {

        Interaction AreaLight::sample(float u) const {
            Interaction ret;
            // todo
            return ret;
        }

        LightLiSample AreaLight::Li(LightLiSample lls) const {
            // todo
            return lls;
        }

        float AreaLight::PDF_Li(const Interaction &ref_p, float3 wi) const {
            //todo
            return 0;
        }

        float3 AreaLight::power() const {
            return (_two_sided ? _2Pi : Pi) * _L / _inv_area;
        }

        float3 AreaLight::L(const Interaction &ref_p) const {
            // todo
            return {};
        }

        std::string AreaLight::to_string() const {
            return string_printf("light Base : %s, L : %s",
                                 _to_string().c_str(),
                                 _L.to_string().c_str());
        }

        AreaLight AreaLight::create(const LightConfig &config) {
            return AreaLight(config.instance_idx, config.emission);
        }
    } //luminous::render
} // luminous::render