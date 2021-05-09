//
// Created by Zero on 2021/5/8.
//

#include "interaction.h"
#include "render/include/shader_data.h"
#include "render/lights/light.h"

namespace luminous {
    inline namespace render {
        Spectrum SurfaceInteraction::Le(float3 w) const {
            DCHECK(light->isa<AreaLight>());
            return light != nullptr ? light->get<AreaLight>()->L(*this, w) : 0.f;
        }

        SurfaceInteraction RadiancePRD::get_surface_interaction() const {
            return hit_group_data->compute_surface_interaction(closest_hit);
        }
    } // luminous::render
} // luminous