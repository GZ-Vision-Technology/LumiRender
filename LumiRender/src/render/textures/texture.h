//
// Created by Zero on 2021/4/16.
//


#pragma once

#include "graphics/math/common.h"
#include "constant_texture.h"
#include "image_texture.h"
#include "graphics/lstd/lstd.h"

namespace luminous {
    inline namespace render {
        using lstd::Variant;
        class Texture : public Variant<ConstantTexture, ImageTexture> {
        private:
            using Variant::Variant;
        public:
            GEN_BASE_NAME(Texture)

            Texture() {}

            GEN_NAME_AND_TO_STRING_FUNC

            NDSC_XPU float4 eval(const TextureEvalContext &tec) {
                LUMINOUS_VAR_DISPATCH(eval, tec)
            }

            NDSC_XPU PixelFormat pixel_format() const {
                LUMINOUS_VAR_DISPATCH(pixel_format)
            }

            XPU void set_mapping(const TextureMapping2D &mapping) {
                LUMINOUS_VAR_DISPATCH(set_mapping, mapping)
            }

            static Texture create(const TextureConfig &config) {
                return detail::create<Texture>(config);
            }
        };
    }
}