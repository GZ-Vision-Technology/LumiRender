//
// Created by Zero on 18/12/2021.
//


#pragma once

#include "render/textures/texture.h"
#include "render/scattering/bsdf_wrapper.h"
#include "parser/config.h"
#include "core/concepts.h"

namespace luminous {
    inline namespace render {
        class FakeMetalMaterial {
        public:
            DECLARE_REFLECTION(FakeMetalMaterial)

        private:
            index_t _color_idx{};
            index_t _roughness_idx{};
            bool _remapping_roughness{};
        public:
            explicit FakeMetalMaterial(index_t color_idx, index_t roughness_idx, bool remapping)
                    : _color_idx(color_idx), _roughness_idx(roughness_idx), _remapping_roughness(remapping) {}

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

            CPU_ONLY(explicit FakeMetalMaterial(const MaterialConfig &mc)
                    : FakeMetalMaterial(mc.color.tex_idx(), mc.roughness.tex_idx(), mc.remapping_roughness) {})
        };

        class MetalMaterial {
        public:
            DECLARE_REFLECTION(MetalMaterial)

        private:
            index_t _eta_idx{};
            index_t _k_idx{};
            index_t _roughness_idx{};
            bool _remapping_roughness{};
        public:
            explicit MetalMaterial(index_t eta_idx, index_t k_idx, index_t roughness_idx, bool remapping)
                    : _eta_idx(eta_idx), _k_idx(k_idx), _roughness_idx(roughness_idx),
                      _remapping_roughness(remapping) {}

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

            CPU_ONLY(explicit MetalMaterial(const MaterialConfig &mc)
                    : MetalMaterial(mc.eta.tex_idx(), mc.k.tex_idx(),
                                    mc.roughness.tex_idx(), mc.remapping_roughness) {})
        };
    }
}