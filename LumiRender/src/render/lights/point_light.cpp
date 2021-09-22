//
// Created by Zero on 2021/4/7.
//

#include "point_light.h"
#include "render/include/creator.h"
#include "render/include/trace.h"

namespace luminous {
    inline namespace render {

        SurfaceInteraction PointLight::sample(LightLiSample *lls, float2 u, const SceneData *scene_data) const {
            SurfaceInteraction ret;
            ret.pos = _pos;
            return ret;
        }

        float PointLight::PDF_Li(const Interaction &ref_p, const SurfaceInteraction &p_light,
                                 float3 wi, const SceneData *data) const {
            return 0;
        }

        Spectrum PointLight::power() const {
            return 4 * Pi * _intensity;
        }

        void PointLight::print() const {
            printf("type:PointLight,L:(%f,%f,%f)\n", _intensity.x, _intensity.y, _intensity.z);
        }

        LightLiSample PointLight::Li(LightLiSample lls, const SceneData *data) const {
            float3 wi = lls.p_light.pos - lls.p_ref.pos;
            lls.L = _intensity / length_squared(wi);
            lls.PDF_dir = 0;
            lls.wi = normalize(wi);
            return lls;
        }
    } // luminous::render
} // luminous