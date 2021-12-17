//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "render/textures/texture.h"
#include "render/scattering/bsdf_wrapper.h"
#include "render/include/config.h"
#include "core/concepts.h"


namespace luminous {
    inline namespace render {
        class MatteMaterial : BASE_CLASS() {
        public:
            REFL_CLASS(MatteMaterial)
        private:
            index_t _color_idx{};
            index_t _sigma{};
        public:
            explicit MatteMaterial(index_t r, index_t sigma) : _color_idx(r), _sigma(sigma) {}

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

            CPU_ONLY(explicit MatteMaterial(const MaterialConfig &mc)
                             :MatteMaterial(mc.color_tex.tex_idx,
                                            mc.sigma_tex.tex_idx) {})
        };
    }
}