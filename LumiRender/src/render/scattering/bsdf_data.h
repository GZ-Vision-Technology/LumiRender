//
// Created by Zero on 16/12/2021.
//


#pragma once

#include "base_libs/optics/rgb.h"

namespace luminous {
    inline namespace render {
        struct BSDFCommonData {
            float4 color{0.f};

            LM_XPU explicit BSDFCommonData(float4 color)
                    : color(color) {}
        };

        struct GlassData : public BSDFCommonData {
            float eta{};
            LM_XPU GlassData(float4 color, float eta)
                    : BSDFCommonData(color), eta(eta) {}
        };

        struct OrenNayarData : public BSDFCommonData {
            float A{};
            float B{};

            LM_XPU OrenNayarData(float4 color, float sigma)
                    : BSDFCommonData(color) {
                sigma = radians(sigma);
                float sigma2 = sqr(sigma);
                A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
                B = 0.45f * sigma2 / (sigma2 + 0.09f);
            }
        };
    }
}