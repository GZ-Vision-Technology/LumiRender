//
// Created by Zero on 2021/4/18.
//


#pragma once

#include "gpu/framework/helper/cuda.h"
#include "base_libs/math/common.h"
#include "texture_base.h"

#ifndef __CUDACC__

#include "cpu/texture/mipmap.h"

#endif

namespace luminous {
    inline namespace render {
        class ImageTexture : public TextureBase {

            DECLARE_REFLECTION(ImageTexture, TextureBase)

        private:
#ifdef CUDA_SUPPORT
            using TypeHandle = CUtexObject;
#else
            using TypeHandle = uint64_t ;
#endif
            TypeHandle _handle{0};
        public:
            CPU_ONLY(explicit ImageTexture(const MaterialAttrConfig &config)
                    : ImageTexture((ImageTexture::TypeHandle) config.handle, config.pixel_format) {})

            ImageTexture(TypeHandle handle, PixelFormat pixel_format)
                    : TextureBase(pixel_format), _handle(handle) {}

#ifndef __CUDACC__

            LM_NODISCARD luminous::float4 eval_on_cpu(const TextureEvalContext &tec) const {
                const MIPMap *mipmap = reinterpret_cast<const MIPMap *>(_handle);
                return mipmap->lookup(tec.uv);
            }

#endif

            LM_ND_XPU luminous::float4 eval(const TextureEvalContext &tec) const {
#ifdef IS_GPU_CODE
                switch (_pixel_format) {
                    case utility::PixelFormat::RGBA8U:
                    case utility::PixelFormat::RGBA32F: {
                        auto val = tex2D<::float4>(_handle, tec.uv[0], tec.uv[1]);
                        return make_float4(val.x, val.y, val.z, val.w);
                    }
                    case utility::PixelFormat::R8U:
                    case utility::PixelFormat::R32F: {
                        auto val = tex2D<float>(_handle, tec.uv[0], tec.uv[1]);
                        return make_float4(val);
                    }
                    case utility::PixelFormat::RG8U:
                    case utility::PixelFormat::RG32F: {
                        auto val = tex2D<::float2>(_handle, tec.uv[0], tec.uv[1]);
                        return make_float4(val.x, val.y, 0, 0);
                    }
                }
#else
                return eval_on_cpu(tec);
#endif
            }

            LM_ND_XPU float4 eval(float2 uv) const {
                return eval(TextureEvalContext(uv));
            }

            LM_XPU void print() const {
                printf("ImageTexture\n");
            }

            GEN_STRING_FUNC({
                                LUMINOUS_TO_STRING("name: %s", type_name(this));
                            })


        };
    }
}