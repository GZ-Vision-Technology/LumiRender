//
// Created by Zero on 2021/4/16.
//


#pragma once

#include "base_libs/math/common.h"
#include "constant_texture.h"
#include "image_texture.h"
#include "base_libs/lstd/lstd.h"

namespace luminous {
    inline namespace render {
        using lstd::Variant;
        class Texture : public Variant<ConstantTexture, ImageTexture> {
        private:
            using Variant::Variant;
        public:
            GEN_BASE_NAME(Texture)

            Texture() {}

            GEN_TO_STRING_FUNC

            XPU void print() const {
                LUMINOUS_VAR_DISPATCH(print)
            }

            NDSC_XPU float4 eval(const TextureEvalContext &tec) const {
                LUMINOUS_VAR_DISPATCH(eval, tec)
            }

            NDSC_XPU float4 eval(float2 uv) const {
                return eval(TextureEvalContext(uv));
            }

            NDSC_XPU PixelFormat pixel_format() const {
                LUMINOUS_VAR_DISPATCH(pixel_format)
            }

            XPU void set_mapping(const TextureMapping2D &mapping) {
                LUMINOUS_VAR_DISPATCH(set_mapping, mapping)
            }

            CPU_ONLY(static Texture create(const TextureConfig &config) {
                return detail::create<Texture>(config);
            })
        };
    }
}