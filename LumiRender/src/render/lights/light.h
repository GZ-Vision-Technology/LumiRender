//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "graphics/geometry/common.h"
#include "graphics/optics/common.h"
#include "render/include/interaction.h"

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
            const LightType _type;
        public:
            XPU LightBase(LightType type) : _type(type) {}

            NDSC_XPU LightType type() const {
                return _type;
            }

            NDSC_XPU bool is_delta() const {
                return _type == LightType::DeltaDirection || _type == LightType::DeltaPosition;
            }

            NDSC std::string _to_string() const {
                return string_printf("type : %d", (int)_type);
            }
        };
    }
}