//
// Created by Zero on 2021/4/16.
//


#pragma once

#include "graphics/math/common.h"
#include "render/include/interaction.h"
#include "graphics/lstd/variant.h"
#include "render/include/creator.h"
#include "util/image_base.h"
#include "render/include/config.h"

namespace luminous {
    inline namespace render {

        class UVMapping2D {
        private:
            float _su, _sv, _du, _dv;
        public:
            UVMapping2D(float su = 1, float sv = 1, float du = 0, float dv = 0)
                    : _su(su), _sv(sv), _du(du), _dv(dv) {}

            std::string to_string() const {
                return string_printf("%s,su:%f, sv:%f,du:%f,dv:%f",
                                     type_name(this), _su, _sv, _du, _dv);
            }

            XPU float2 map(const TextureEvalContext &ctx, float2 *dst_dx, float2 *dst_dy) const {
                if (dst_dx) { *dst_dx = float2(_su * ctx.du_dx, _sv * ctx.dv_dx); }
                if (dst_dy) { *dst_dy = float2(_su * ctx.du_dy, _sv * ctx.dv_dy); }
                return make_float2(_su * ctx.uv[0] + _du, _sv * ctx.uv[1] + _dv);
            }

            static UVMapping2D create(const TextureMappingConfig tmc) {
                return UVMapping2D(tmc.su, tmc.sv, tmc.du, tmc.dv);
            }
        };

        using lstd::Variant;

        class TextureMapping2D : public Variant<UVMapping2D> {
        private:
            using Variant::Variant;
        public:
            GEN_BASE_NAME(TextureMapping2D)

            XPU float2 map(const TextureEvalContext &ctx, float2 *dst_dx, float2 *dst_dy) const {
                LUMINOUS_VAR_DISPATCH(map, ctx, dst_dx, dst_dy)
            }

            GEN_NAME_AND_TO_STRING_FUNC

            static TextureMapping2D create(const TextureMappingConfig &tmc) {
                return detail::create<TextureMapping2D>(tmc);
            }
        };

        class TextureBase {
        protected:
            PixelFormat _pixel_format;
            TextureMapping2D _mapping;
        public:
            XPU TextureBase() = default;

            XPU TextureBase(PixelFormat pixel_format)
                    : _pixel_format(pixel_format) {}

            NDSC_XPU_INLINE PixelFormat pixel_format() const {
                return _pixel_format;
            }

            XPU_INLINE void set_mapping(const TextureMapping2D &mapping) {
                _mapping = mapping;
            }
        };

    } //luminous::render
} //luminous