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
            index_t R{};
        public:
            explicit MatteMaterial(index_t r) : R(r) {}

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

            CPU_ONLY(explicit MatteMaterial(const MaterialConfig &mc)
                             :MatteMaterial(mc.diffuse_tex.tex_idx) {})
        };
    }
}