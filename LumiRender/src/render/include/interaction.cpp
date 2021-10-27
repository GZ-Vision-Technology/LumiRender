//
// Created by Zero on 2021/5/8.
//

#include "interaction.h"
#include "render/scene/scene_data.h"
#include "render/lights/area_light.h"
#include "render/lights/light.h"
#include "render/materials/material.h"

namespace luminous {
    inline namespace render {
        Spectrum SurfaceInteraction::Le(float3 w,const SceneData *scene_data) const {
            return has_emission() ? (*light->get<AreaLight*>())->radiance(*this, w, scene_data) : 0.f;
        }

        SurfaceInteraction PerRayData::compute_surface_interaction(Ray ray) const {
            auto si = scene_data()->compute_surface_interaction(hit_point);
            si.init_BSDF(scene_data());
            si.wo = normalize(-ray.direction());
            return si;
        }

        lstd::optional<BSDF> SurfaceInteraction::get_BSDF(const SceneData *scene_data) const {
            if (!has_material()) {
                return {};
            }
            auto bsdf = material->get_BSDF(*this, scene_data);
            return lstd::optional<BSDF>(bsdf);
        }
    } // luminous::render
} // luminous