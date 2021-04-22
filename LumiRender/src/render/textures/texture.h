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
        template<typename T>
        class Texture : public Variant<ConstantTexture<T>, GPUImageTexture<T>> {
//        private:
//            using Variant::Variant;
        public:
            using value_type = T;

            GEN_BASE_NAME(Texture<T>)

            GEN_NAME_AND_TO_STRING_FUNC

            XPU T eval(const TextureEvalContext &tec) {
                LUMINOUS_VAR_DISPATCH(eval, tec)
            }

            XPU void set_mapping(const TextureMapping2D &mapping) {
                LUMINOUS_VAR_DISPATCH(set_mapping, mapping)
            }

            static Texture<T> create(const TextureConfig<T> &config);
        };

        template<typename T>
        Texture<T> Texture<T>::create(const TextureConfig<T> &config) {
            return detail::create<Texture<T>>(config);
        }
    }
}