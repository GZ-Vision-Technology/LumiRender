//
// Created by Zero on 2021/4/16.
//


#pragma once

#include "graphics/math/common.h"
#include "constant_texture.h"
#include "image_texture.h"
#include "graphics/lstd/variant.h"

namespace luminous {
    inline namespace render {
        using lstd::Variant;
        template<typename T>
        class Texture : Variant<ConstantTexture<T>> {
        public:
            using value_type = T;

            GEN_NAME_AND_TO_STRING_FUNC

            XPU T eval(const TextureEvalContext &tec) {
                LUMINOUS_VAR_DISPATCH(eval, tec)
            }

            XPU void set_mapping(const TextureMapping2D &mapping) {
                LUMINOUS_VAR_DISPATCH(set_mapping, mapping)
            }


        };
    }
}