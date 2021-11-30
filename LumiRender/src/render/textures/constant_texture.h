//
// Created by Zero on 2021/2/17.
//


#pragma once

#include "base_libs/math/common.h"

#include "texture_base.h"

namespace luminous {
    inline namespace render {
        class ConstantTexture : public TextureBase {
        private:
            const float4 _val;
        public:
            CPU_ONLY(explicit ConstantTexture(const TextureConfig &config)
                    : ConstantTexture(PixelFormat::RGBA32F, config.val) {})

            ConstantTexture(PixelFormat pixel_format, float4 val)
                    : TextureBase(pixel_format), _val(val) {}

            LM_ND_XPU float4 eval(const TextureEvalContext &tec) const {
                return _val;
            }

            LM_XPU void print() const {
                printf("ConstantTexture: %f,%f,%f,%f\n", _val.x, _val.y, _val.z, _val.w);
            }

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("name: %s", type_name(this));
                            })


        };
    }
}