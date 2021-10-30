//
// Created by Zero on 2021/4/7.
//


#pragma once

#include "light_base.h"
#include "render/include/config.h"

namespace luminous {
    inline namespace render {

        class SpotLight : BASE_CLASS(LightBase) {
        public:
            REFL_CLASS(SpotLight)
        private:
            float3 _pos;
            // center direction in world space
            float3 _axis;
            // decay from inner ring
            float _cos_theta_i;
            // decay end with outer ring
            float _cos_theta_o;
            float3 _intensity;
        public:
            CPU_ONLY(SpotLight(const LightConfig &config)
                             :SpotLight(config.position,
                             config.intensity,
                             std::cos(config.theta_i),
                             std::cos(config.theta_o)) {})

            SpotLight(float3 pos, float3 intensity, float theta_i, float theta_o)
                    : BaseBinder<LightBase>(LightType::DeltaPosition),
                      _pos(pos),
                      _intensity(intensity),
                      _cos_theta_i(cos(radians(theta_i))),
                      _cos_theta_o(cos(radians(theta_o))) {}

            LM_ND_XPU SurfaceInteraction sample(LightLiSample *lls, float2 u, const SceneData *scene_data) const;

            LM_ND_XPU LightLiSample Li(LightLiSample lls, const SceneData *data) const;

            /**
             * @param w_world : unit vector in world space
             * @return
             */
            LM_ND_XPU float fall_off(float3 w_world) const;

            LM_ND_XPU float PDF_Li(const LightSampleContext &ctx, const AreaLightEvalContext &p_light,
                                   float3 wi, const SceneData *data) const;

            LM_ND_XPU Spectrum power() const;

            LM_XPU void print() const;

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("light Base : %s, name : %s ,intensity : %s",
                                                   LightBase::to_string().c_str(),
                                                   type_name(this),
                                                   _intensity.to_string().c_str());
                            })

        };

    } // luminous::render
} // luminous