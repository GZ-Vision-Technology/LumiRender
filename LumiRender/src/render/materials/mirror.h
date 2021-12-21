//
// Created by Zero on 17/12/2021.
//


#pragma once

#include "render/textures/texture.h"
#include "render/scattering/bsdf_wrapper.h"
#include "render/include/config.h"
#include "core/concepts.h"

namespace luminous {
    inline namespace render {
        class MirrorMaterial : BASE_CLASS() {
        public:
            REFL_CLASS(MirrorMaterial)

        private:
            index_t _color_idx{};
        public:
            explicit MirrorMaterial(index_t color_idx) : _color_idx(color_idx) {}

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

            CPU_ONLY(explicit MirrorMaterial(const MaterialConfig &mc)
                    : MirrorMaterial(mc.color_tex.tex_idx()) {})
        };
    }
}