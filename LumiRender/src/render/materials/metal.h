//
// Created by Zero on 18/12/2021.
//


#pragma once

#include "render/textures/texture.h"
#include "render/scattering/bsdf_wrapper.h"
#include "parser/config.h"
#include "core/concepts.h"
#include "attr.h"

namespace luminous {
    inline namespace render {
        class FakeMetalMaterial {
        public:
            DECLARE_REFLECTION(FakeMetalMaterial)
        private:
            Attr3D _color{};
            Attr2D _roughness{};
            bool _remapping_roughness{};
        public:

            FakeMetalMaterial(Attr3D color, Attr2D roughness, bool remapping)
                    : _color(color), _roughness(roughness), _remapping_roughness(remapping) {}

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

            CPU_ONLY(explicit FakeMetalMaterial(const MaterialConfig &mc)
                    : FakeMetalMaterial(Attr3D(mc.color), Attr2D(mc.roughness), mc.remapping_roughness) {})
        };

        class MetalMaterial {
        public:
            DECLARE_REFLECTION(MetalMaterial)

        private:
            Attr3D _eta{};
            Attr3D _k{};
            Attr2D _roughness{};
            bool _remapping_roughness{};
        public:
            explicit MetalMaterial(Attr3D eta, Attr3D k, Attr2D roughness, bool remapping)
                    : _eta(eta), _k(k), _roughness(roughness),
                      _remapping_roughness(remapping) {}

            LM_ND_XPU BSDFWrapper get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const;

            CPU_ONLY(explicit MetalMaterial(const MaterialConfig &mc)
                    : MetalMaterial(Attr3D(mc.eta), Attr3D(mc.k),
                                    Attr2D(mc.roughness), mc.remapping_roughness) {})
        };
    }
}