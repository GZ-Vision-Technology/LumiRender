//
// Created by Zero on 2021/4/29.
//


#pragma once

#include "graphics/math/common.h"

namespace luminous {
    inline namespace render {
        // BxDFFlags Definition
        enum BxDFFlags {
            Unset = 0,
            Reflection = 1 << 0,
            Transmission = 1 << 1,
            Diffuse = 1 << 2,
            Glossy = 1 << 3,
            Specular = 1 << 4,
            // Composite _BxDFFlags_ definitions
            DiffuseReflection = Diffuse | Reflection,
            DiffuseTransmission = Diffuse | Transmission,
            GlossyReflection = Glossy | Reflection,
            GlossyTransmission = Glossy | Transmission,
            SpecularReflection = Specular | Reflection,
            SpecularTransmission = Specular | Transmission,
            All = Diffuse | Glossy | Specular | Reflection | Transmission
        };
    }
}