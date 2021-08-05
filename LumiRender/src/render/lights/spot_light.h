//
// Created by Zero on 2021/4/7.
//


#pragma once

#include "light_base.h"
#include "render/include/config.h"

namespace luminous {
    inline namespace render {

        class SpotLight : public LightBase {
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
            SpotLight(float3 pos, float3 intensity, float theta_i, float theta_o)
                    : LightBase(LightType::DeltaPosition),
                      _pos(pos),
                      _intensity(intensity),
                      _cos_theta_i(cos(radians(theta_i))),
                      _cos_theta_o(cos(radians(theta_o))) {}

            NDSC_XPU SurfaceInteraction sample(float2 u, const SceneData *hit_group_data) const;

            NDSC_XPU LightLiSample Li(LightLiSample lls) const;

            /**
             * @param w_world : unit vector in world space
             * @return
             */
            NDSC_XPU float fall_off(float3 w_world) const;

            NDSC_XPU float PDF_Li(const Interaction &ref_p, const SurfaceInteraction &p_light) const;

            NDSC_XPU Spectrum power() const;

            XPU void print() const;

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("light Base : %s, name : %s ,intensity : %s",
                                                   LightBase::to_string().c_str(),
                                                   type_name(this),
                                                   _intensity.to_string().c_str());
                            })

            CPU_ONLY(static SpotLight create(const LightConfig &config);)
        };

    } // luminous::render
} // luminous