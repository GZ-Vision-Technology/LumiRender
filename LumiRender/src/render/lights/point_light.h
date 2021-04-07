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

            NDSC_XPU float3 sample_Li(DirectSamplingRecord *rcd, float2 u) const;

            NDSC_XPU float PDF_Li(const DirectSamplingRecord &rcd) const;

            NDSC_XPU float3 power() const;

            NDSC std::string to_string() const;
        };
    } // luminous::render
} // luminous