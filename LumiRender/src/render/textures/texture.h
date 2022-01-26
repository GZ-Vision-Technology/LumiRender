//
// Created by Zero on 2021/4/16.
//


#pragma once

#include "base_libs/math/common.h"
#include "image_texture.h"
#include "base_libs/lstd/lstd.h"

namespace luminous {
    inline namespace render {
        using lstd::Variant;
        class Texture : public Variant<ImageTexture> {

            DECLARE_REFLECTION(Texture, ImageTexture)

        private:
            using Variant::Variant;
        public:
            Texture() = default;

            GEN_TO_STRING_FUNC

            LM_XPU void print() const {
                LUMINOUS_VAR_DISPATCH(print)
            }

            LM_ND_XPU float4 eval(const TextureEvalContext &tec) const {
//                return get<ImageTexture>()->eval(tec);
                // todo a strange bug
                LUMINOUS_VAR_DISPATCH(eval, tec)
            }

            LM_ND_XPU float4 eval(float2 uv) const {
                return eval(TextureEvalContext(uv));
            }

            LM_ND_XPU PixelFormat pixel_format() const {
                LUMINOUS_VAR_DISPATCH(pixel_format)
            }

            LM_ND_XPU int channel_num() const {
                LUMINOUS_VAR_DISPATCH(channel_num);
            }

            CPU_ONLY(static Texture create(const MaterialAttrConfig &config) {
                return detail::create<Texture>(config);
            })
        };
    }
}