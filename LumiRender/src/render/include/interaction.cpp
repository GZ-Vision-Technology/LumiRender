//
// Created by Zero on 2021/5/8.
//

#include "interaction.h"
#include "render/include/shader_data.h"
#include "render/lights/light.h"
#include "render/bxdfs/bsdf.h"
#include "render/materials/material.h"

namespace luminous {
    inline namespace render {
        Spectrum SurfaceInteraction::Le(float3 w) const {
            return has_emission() ? light->get<AreaLight>()->L(*this, w) : 0.f;
        }

        const HitGroupData *PerRayData::hit_group_data() const {
            return reinterpret_cast<const HitGroupData *>(data);
        }

        const MissData *PerRayData::miss_data() const {
            return reinterpret_cast<const MissData *>(data);
        }

        SurfaceInteraction PerRayData::get_surface_interaction() const {
            auto si = hit_group_data()->compute_surface_interaction(closest_hit);
            si.wo = normalize(-ray_dir);
            return si;
        }

        lstd::optional<BSDF> SurfaceInteraction::get_BSDF(const HitGroupData *hit_group_data) const {
            if (!has_material()) {
                return {};
            }
            auto bsdf = material->get_BSDF(*this, hit_group_data);
            return lstd::optional<BSDF>(bsdf);
        }
    } // luminous::render
} // luminous