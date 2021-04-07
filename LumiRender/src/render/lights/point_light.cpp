//
// Created by Zero on 2021/4/7.
//

#include "point_light.h"

namespace luminous {
    inline namespace render {

        float3 PointLight::sample_Li(DirectSamplingRecord *rcd, float2 u) const {
            //todo
            return _intensity;
        }

        float PointLight::PDF_Li(const DirectSamplingRecord &rcd) const {
            return rcd.PDF_dir;
        }

        float3 PointLight::power() const {
            return 4 * Pi * _intensity;
        }

        std::string PointLight::to_string() const {
            return string_printf("light Base : %s, intensity : %s",
                                 _to_string().c_str(),
                                 _intensity.to_string().c_str());
        }
    } // luminous::render
} // luminous