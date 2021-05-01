//
// Created by Zero on 2021/4/30.
//


#pragma once

#include "matte.h"
#include "ai_material.h"
#include "graphics/lstd/variant.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;
        class Material : public Variant<MatteMaterial, AssimpMaterial> {
            using Variant::Variant;
        public:
            NDSC_XPU BSDF get_BSDF(TextureEvalContext tec, const HitGroupData* hit_group_data) {
                LUMINOUS_VAR_DISPATCH(get_BSDF, tec, hit_group_data)
            }
        };

    } // luminous::render
} // luminous

