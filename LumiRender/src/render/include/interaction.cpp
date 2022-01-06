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
        Spectrum SurfaceInteraction::Le(float3 w, const SceneData *scene_data) const {
            return has_emission() ?
                   light->as<AreaLight>()->radiance(LightEvalContext{*this}, w, scene_data) :
                   0.f;
        }

        SurfaceInteraction HitContext::compute_surface_interaction(Ray ray) const {
            return compute_surface_interaction(normalize(-ray.direction()));
        }

        SurfaceInteraction HitContext::compute_surface_interaction(float3 wo) const {
            auto si = scene_data()->compute_surface_interaction(hit_info);
            si.wo = wo;
            return si;
        }

        bool HitContext::has_emission() const {
            return data->has_emission(hit_info.instance_id);
        }

        bool HitContext::has_material() const {
            return data->has_material(hit_info.instance_id);
        }

        SurfacePoint HitContext::surface_point() const {
            float3 pos, ng;
            data->fill_attribute(hit_info, &pos, &ng);
            return {pos, ng};
        }

        GeometrySurfacePoint HitContext::geometry_surface_point() const {
            float3 pos, ng;
            float2 uv;
            data->fill_attribute(hit_info, &pos, &ng, &uv);
            return {pos, ng, uv};
        }

        LightEvalContext HitContext::compute_light_eval_context() const {
            return data->compute_light_eval_context(hit_info);
        }

        const Light *HitContext::light() const {
            return data->get_light(hit_info.instance_id);
        }

        const Material *HitContext::material() const {
            return data->get_material(hit_info.instance_id);
        }

        float HitContext::compute_prim_PMF() const {
            return data->compute_prim_PMF(hit_info);
        }

        BSDFWrapper SurfaceInteraction::compute_BSDF(const SceneData *scene_data) const {
            LM_ASSERT(bool(material), "material is nullptr!\n");
            BSDFWrapper bsdf_wrapper = material->get_BSDF(*this, scene_data);
            s_uvn.set_frame(bsdf_wrapper.shading_frame());
            return bsdf_wrapper;
        }
    } // luminous::render
} // luminous