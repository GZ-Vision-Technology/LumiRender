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
            return has_emission() ? light->as<AreaLight>()->radiance(LightEvalContext{*this}, w, scene_data) : 0.f;
        }

        SurfaceInteraction HitContext::compute_surface_interaction(Ray ray) const {
            auto si = scene_data()->compute_surface_interaction(hit_info);
            si.init_BSDF(scene_data());
            si.wo = normalize(-ray.direction());
            return si;
        }

        bool HitContext::has_emission() const {
            return data->has_emission(hit_info.instance_id);
        }

        bool HitContext::has_material() const {
            return data->has_material(hit_info.instance_id);
        }

        std::pair<float3, float3> HitContext::compute_geometry() const {
            float3 pos, ng;
            data->fill_attribute(hit_info, &pos, &ng);
            return {pos, ng};
        }

        lstd::optional<BSDF> SurfaceInteraction::get_BSDF(const SceneData *scene_data) const {
            if (!has_material()) {
                return {};
            }
            auto bsdf = material->get_BSDF(*this, scene_data);
            return {bsdf};
        }
    } // luminous::render
} // luminous