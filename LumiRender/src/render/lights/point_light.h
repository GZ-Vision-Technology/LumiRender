//
// Created by Zero on 2021/4/7.
//


#pragma once

#include "light_base.h"
#include "render/include/config.h"

namespace luminous {
    inline namespace render {
        class PointLight : BASE_CLASS(LightBase), public ICreator<PointLight> {
        public:
            REFL_CLASS(PointLight)
        private:
            float3 _pos;
            Spectrum _intensity;
        public:
            CPU_ONLY(explicit PointLight(const LightConfig &config)
                    : PointLight(config.position, config.intensity) {})

            PointLight(float3 pos, float3 intensity)
                    : BaseBinder<LightBase>(LightType::DeltaPosition),
                      _pos(pos),
                      _intensity(intensity) {}

            NDSC_XPU SurfaceInteraction sample(LightLiSample *lls, float2 u, const SceneData *scene_data) const;

            NDSC_XPU LightLiSample Li(LightLiSample lls, const SceneData *data) const;

            NDSC_XPU float PDF_Li(const Interaction &ref_p, const SurfaceInteraction &p_light,
                                  float3 wi, const SceneData *data) const;

            NDSC_XPU Spectrum power() const;

            XPU void print() const;

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("light Base : %s, name : %s ,intensity : %s",
                                                   LightBase::to_string().c_str(),
                                                   type_name(this),
                                                   _intensity.to_string().c_str());
                            })
        };
    } // luminous::render
} // luminous