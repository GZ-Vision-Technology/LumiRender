//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "graphics/geometry/common.h"
#include "graphics/optics/common.h"

namespace luminous {
    inline namespace render {

        // LightType Definition
        enum class LightType {
            DeltaPosition,
            DeltaDirection,
            Area,
            Infinite
        };

        enum class LightSamplingMode {
            WithMIS,
            WithoutMIS
        };

        class LightBase {
        protected:
            LightType _type;
        public:
            XPU LightBase(LightType type) : _type(type) {}

            NDSC_XPU Spectrum L(float3 p, float3 n, float2, float3 wi) const {
                return Spectrum(0,0,0);
            }

            NDSC_XPU LightType type() const {
                return _type;
            }

            NDSC std::string _to_string() const {
                return string_printf("type : %d", (int)_type);
            }
        };
    }
}