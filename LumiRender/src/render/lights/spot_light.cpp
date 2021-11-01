//
// Created by Zero on 2021/4/8.
//


#include "spot_light.h"
#include "core/refl/factory.h"

namespace luminous {
    inline namespace render {

        LightEvalContext SpotLight::sample(LightLiSample *lls, float2 u, const SceneData *scene_data) const {
            auto lec = LightEvalContext{_pos, make_float3(0.f), make_float2(0.f), 0};
            lls->set_sample_result(0, lec, normalize(_pos - lls->lsc.pos));
            return lec;
        }

        float SpotLight::fall_off(float3 w_world) const {
            float cos_theta = dot(w_world, _axis);
            if (cos_theta > _cos_theta_i) {
                return 1;
            } else if (cos_theta <= _cos_theta_o) {
                return 0;
            }
            float delta = (cos_theta - _cos_theta_o) / (_cos_theta_i - _cos_theta_o);
            return sqr(sqr(delta));
        }

        void SpotLight::print() const {
            printf("type:SpotLight,L:(%f,%f,%f)\n", _intensity.x, _intensity.y, _intensity.z);
        }

        LightLiSample SpotLight::Li(LightLiSample lls, const SceneData *data) const {
            float3 wi = lls.lsc.pos - lls.lec.pos;
            float3 L = _intensity / length_squared(wi);
            float f = fall_off(normalize(wi));
            lls.update_Li(L * f);
            return lls;
        }

        float SpotLight::PDF_Li(const LightSampleContext &ctx, const LightEvalContext &p_light,
                                float3 wi, const SceneData *data) const {
            return 0;
        }

        Spectrum SpotLight::power() const {
            return _intensity * 2.f * Pi * (1 - 0.5f * (_cos_theta_i + _cos_theta_o));
        }

    } // luminous::render
} // luminous