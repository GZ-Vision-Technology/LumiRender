//
// Created by Zero on 2021/5/8.
//

#include "interaction.h"
#include "render/lights/light.h"

namespace luminous {
    inline namespace render {
        Spectrum SurfaceInteraction::Le(float3 w) const {
            DCHECK(light->isa<AreaLight>());
            return light != nullptr ? light->get<AreaLight>()->L(*this, w) : 0.f;
        }
    } // luminous::render
} // luminous