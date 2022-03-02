//
// Created by Zero on 2021/4/7.
//

#include "point_light.h"
#include "core/refl/factory.h"

namespace luminous {
    inline namespace render {

        LightEvalContext PointLight::sample(LightLiSample *lls, float2 u, const SceneData *scene_data) const {
            auto lec = LightEvalContext{_pos, make_float3(0.f), make_float2(0.f), 0};
            lls->set_sample_result(0, lec, normalize(_pos - lls->lsc.pos));
            return lec;
        }

        float PointLight::PDF_Li(const LightSampleContext &ctx, const LightEvalContext &p_light,
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
            float3 wi = lls.lec.pos - lls.lsc.pos;
            lls.set_Li(_intensity / length_squared(wi));
            return lls;
        }

    } // luminous::render
} // luminous