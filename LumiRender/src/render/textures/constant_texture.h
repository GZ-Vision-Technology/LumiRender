//
// Created by Zero on 2021/2/17.
//


#pragma once

#include "graphics/math/common.h"

#include "texture_base.h"

namespace luminous {
    inline namespace render {
        class ConstantTexture : public TextureBase {
        private:
            const float4 _val;
        public:
            ConstantTexture(PixelFormat pixel_format, float4 val)
                : TextureBase(pixel_format), _val(val) {}

            XPU float4 eval(const TextureEvalContext &tec) const {
                return _val;
            }

            GEN_STRING_FUNC({
                LUMINOUS_TO_STRING("name: %s", type_name(this));
            })

            static ConstantTexture create(const TextureConfig &config) {
                return ConstantTexture(config.pixel_format, config.val);
            }
        };
    }
}