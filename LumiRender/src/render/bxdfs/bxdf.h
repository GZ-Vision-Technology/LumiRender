//
// Created by Zero on 2021/4/29.
//


#pragma once

#include "diffuse.h"
#include "graphics/lstd/variant.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;
        class BxDF : public Variant<IdealDiffuse> {
            using Variant::Variant;
        public:

        };

    }
}