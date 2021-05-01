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

        };

    } // luminous::render
} // luminous

