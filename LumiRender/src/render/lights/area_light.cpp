//
// Created by Zero on 2021/4/7.
//

#include "area_light.h"

namespace luminous {
    inline namespace render {

        float3 AreaLight::sample_Li(DirectSamplingRecord *rcd, float2 u) const {
            //todo
            return _L;
        }

        float AreaLight::PDF_Li(const DirectSamplingRecord &rcd) const {
            return rcd.PDF_dir;
        }

        float3 AreaLight::power() const {
            return (_two_sided ? _2Pi : Pi) * _L / _inv_area;
        }

        float3 AreaLight::L(const DirectSamplingRecord &rcd) const {
            return (_two_sided || rcd.cos_target_theta() > 0) ? _L : make_float3(0.f);
        }

        std::string AreaLight::to_string() const {
            return string_printf("light Base : %s, L : %s",
                                 _to_string().c_str(),
                                 _L.to_string().c_str());
        }
    } //luminous::render
} // luminous::render