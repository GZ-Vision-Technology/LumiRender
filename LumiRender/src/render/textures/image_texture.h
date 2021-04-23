//
// Created by Zero on 2021/4/18.
//


#pragma once

#include "cuda.h"
#include "graphics/math/common.h"
#include "texture_base.h"

namespace luminous {
    inline namespace render {
        template<typename T>
        class ImageTexture : public TextureBase {
        private:
            CUtexObject _handle{0};
        public:
            ImageTexture(CUtexObject handle)
                    : _handle(handle) {}

            XPU T eval(const TextureEvalContext &tec) {
                return T();
            }

            std::string to_string() const {
                LUMINOUS_TO_STRING("name: %s", name().c_str());
            }

            static ImageTexture <T> create(const TextureConfig <T> &config) {
                return ImageTexture<T>((CUtexObject)config.handle);
            }
        };
    }
}