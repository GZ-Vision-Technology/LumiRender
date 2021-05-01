//
// Created by Zero on 2021/5/1.
//


#pragma once

#include "render/textures/texture.h"

namespace luminous {
    inline namespace render {
        class AssimpMaterial {
        private:
            index_t _diffuse_idx;
            index_t _specular_idx;
            index_t _normal_idx;
        public:
            AssimpMaterial(index_t diffuse_idx, index_t specular_idx, index_t normal_idx)
                : _diffuse_idx(diffuse_idx),
                _specular_idx(specular_idx),
                _normal_idx(normal_idx) {}
        };
    }
}