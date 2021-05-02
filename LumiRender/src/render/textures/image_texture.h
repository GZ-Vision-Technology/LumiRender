//
// Created by Zero on 2021/4/18.
//


#pragma once

#include "cuda.h"
#include "graphics/math/common.h"
#include "texture_base.h"
#include "vector_types.h"

namespace luminous {
    inline namespace render {
        class ImageTexture : public TextureBase {
        private:
            CUtexObject _handle{0};
        public:
            ImageTexture(CUtexObject handle, PixelFormat pixel_format)
                    : TextureBase(pixel_format), _handle(handle) {}

            XPU luminous::float4 eval(const TextureEvalContext &tec) const {
#ifdef IS_GPU_CODE
                switch (_pixel_format) {
                    case utility::PixelFormat::RGBA8U:
                    case utility::PixelFormat::RGBA32F: {
                        auto val = tex2D<::float4>(_handle, tec.uv[0], 1 - tec.uv[1]);
                        return make_float4(val.x, val.y, val.z, val.w);
                    }
                    case utility::PixelFormat::R8U:
                    case utility::PixelFormat::R32F: {
                        auto val = tex2D<float>(_handle, tec.uv[0], 1 - tec.uv[1]);
                        return make_float4(val);
                    }
                    case utility::PixelFormat::RG8U:
                    case utility::PixelFormat::RG32F: {
                        auto val = tex2D<::float2>(_handle, tec.uv[0], 1 - tec.uv[1]);
                        return make_float4(val.x, val.y, 0, 0);
                    }
                }
#else
                return make_float4(1.0);
#endif
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