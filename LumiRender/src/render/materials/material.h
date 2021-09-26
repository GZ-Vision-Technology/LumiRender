//
// Created by Zero on 2021/4/30.
//


#pragma once

#include "matte.h"
#include "ai_material.h"
#include "base_libs/lstd/variant.h"
#include "core/refl/reflection.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;

        class Material : BASE_CLASS(Variant<MatteMaterial, AssimpMaterial>) {
        public:
            REFL_CLASS(Material)

            using BaseBinder::BaseBinder;
        public:
            GEN_BASE_NAME(Material)

            NDSC_XPU BSDF get_BSDF(const MaterialEvalContext &ctx, const SceneData *scene_data) const {

                LUMINOUS_VAR_DISPATCH(get_BSDF, ctx, scene_data)
            }

            CPU_ONLY(static Material create(const MaterialConfig &mc) {
                return detail::create<Material>(mc);
            })
        };

    } // luminous::render
} // luminous

