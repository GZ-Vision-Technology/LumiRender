//
// Created by Zero on 12/10/2021.
//


#pragma once

#include "base_libs/math/common.h"

namespace luminous {
    inline namespace render {
        struct PixelInfo {
        public:
            Spectrum Li{};
            float3 normal{};
            float3 albedo{};

            LM_ND_XPU PixelInfo operator + (const PixelInfo &other) const {
                return {Li + other.Li, normal + other.normal, albedo + other.albedo};
            }

            LM_XPU PixelInfo& operator += (const PixelInfo &other) {
                Li += other.Li;
                normal += other.normal;
                albedo += other.albedo;
                return *this;
            }

            LM_ND_XPU PixelInfo operator / (float num) const {
                return {Li / num, normal / num, albedo / num};
            }

            LM_XPU PixelInfo& operator /= (float num) {
                Li /= num;
                normal /= num;
                albedo /= num;
                return *this;
            }
        };
    }
}