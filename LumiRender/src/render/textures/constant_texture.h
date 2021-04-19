//
// Created by Zero on 2021/2/17.
//


#pragma once

#include "graphics/math/common.h"

#include "texture_base.h"

namespace luminous {
    inline namespace render {
        template<typename T>
        class ConstantTexture : public TextureBase {
        public:
            using value_type = T;
            using hight_type = typename HighPrecision<T>::type;
        private:
            const T val;
        public:
            GEN_CLASS_NAME(ConstantTexture)

            XPU T eval(const TextureEvalContext &tec) {
                return val;
            }

            std::string to_string() const {
                return string_printf("ConstantTexture val%g", val);
            }

        };
    }
}