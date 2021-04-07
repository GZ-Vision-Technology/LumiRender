//
// Created by Zero on 2021/4/7.
//


#pragma once

#include "light.h"

namespace luminous {
    inline namespace render {
        class PointLight : public LightBase {
        private:
            float3 _pos;
            float3 _intensity;
        public:
            PointLight(float3 pos, float3 intensity)
                    : LightBase(LightType::DeltaPosition),
                      _pos(pos),
                      _intensity(intensity) {}

            GEN_CLASS_NAME(PointLight)

            NDSC_XPU Interaction sample(float u) const;

            NDSC_XPU LightLiSample Li(LightLiSample lls) const;

            NDSC_XPU float PDF_Li(const Interaction &ref_p, float3 wi) const;

            NDSC_XPU float3 power() const;

            NDSC std::string to_string() const;
        };
    } // luminous::render
} // luminous