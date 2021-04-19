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
        class GPUImageTexture : public TextureBase {
        private:
            CUtexObject _handle;

        };
    }
}