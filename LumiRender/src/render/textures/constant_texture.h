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
        private:
            const T _val;
        public:
            ConstantTexture(T val)
                :_val(val){}

            GEN_CLASS_NAME(ConstantTexture<T>)

            XPU T eval(const TextureEvalContext &tec) {
                return _val;
            }

            std::string to_string() const {
                LUMINOUS_TO_STRING("name: %s", name().c_str());
            }

            static ConstantTexture<T> create(const TextureConfig<T> &config) {
                return ConstantTexture<T>(config.val);
            }
        };
    }
}