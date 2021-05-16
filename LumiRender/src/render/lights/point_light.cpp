//
// Created by Zero on 2021/4/7.
//

#include "point_light.h"
#include "render/include/creator.h"
#include "render/include/trace.h"

namespace luminous {
    inline namespace render {

        SurfaceInteraction PointLight::sample(float2 u, const HitGroupData *hit_group_data) const {
            SurfaceInteraction ret;
            ret.pos = _pos;
            return ret;
        }

        float PointLight::PDF_dir(const Interaction &ref_p, const SurfaceInteraction &p_light) const {
            return 0;
        }

        Spectrum PointLight::power() const {
            return 4 * Pi * _intensity;
        }

        void PointLight::print() const {
            printf("type:PointLight,L:(%f,%f,%f)\n", _intensity.x, _intensity.y, _intensity.z);
        }

        std::string PointLight::to_string() const {
            LUMINOUS_TO_STRING("light Base : %s, name : %s ,intensity : %s",
                                 _to_string().c_str(),
                                 type_name(this),
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