//
// Created by Zero on 2021/4/18.
//


#pragma once

#include "cuda.h"
#include "graphics/math/common.h"
#include "texture_base.h"
#include <vector_types.h>"

namespace luminous {
    inline namespace render {
        class ImageTexture : public TextureBase {
        private:
            CUtexObject _handle{0};
        public:
            ImageTexture(CUtexObject handle, PixelFormat pixel_format)
                    : TextureBase(pixel_format), _handle(handle) {}

            XPU luminous::float4 eval(const TextureEvalContext &tec) {
                auto val = tex2D<::float4>(_handle, tec.uv[0], 1 - tec.uv[1]);
                return make_float4(1.0);
            }

            std::string to_string() const {
                LUMINOUS_TO_STRING("name: %s", type_name(this));
            }

            static ImageTexture create(const TextureConfig &config) {
                return ImageTexture ((CUtexObject) config.handle, config.pixel_format);
            }
        };
    }
}