//
// Created by Zero on 2021/6/9.
//


#pragma once

#include "render/textures/texture.h"
#include "render/bxdfs/bsdf.h"
#include "render/include/config.h"
#include "core/concepts.h"

namespace luminous {
    inline namespace render {
        class DielectricMaterial : BASE_CLASS() {
        public:
            REFL_CLASS(DielectricMaterial)

        private:
            index_t Kr{};
            index_t Kt{};
            index_t _roughness{};
            index_t _eta{};
        public:
            explicit DielectricMaterial(index_t kr, index_t kt, index_t roughness, index_t eta)
                    : Kr(kr), Kt(kt), _roughness(roughness), _eta(eta) {}

            LM_ND_XPU BSDF get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

            CPU_ONLY(explicit DielectricMaterial(const MaterialConfig &mc)
                    : DielectricMaterial(mc.Kr_tex.tex_idx,
                                         mc.Kt_tex.tex_idx,
                                         mc.roughness_tex.tex_idx,
                                         mc.eta_tex.tex_idx) {})
        };
    }
}