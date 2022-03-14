//
// Created by Zero on 2021/4/7.
//

#include "common.h"
#include "light.h"
#include "render/include/trace.h"
#include "core/refl/factory.h"
#include "render/scene/scene_data.h"

namespace luminous {
    inline namespace render {


        LightType Light::type() const {
            LUMINOUS_VAR_PTR_DISPATCH(type);
        }

        LightEvalContext Light::sample(LightLiSample *lls, float2 u, const SceneData *scene_data) const {
            LUMINOUS_VAR_PTR_DISPATCH(sample, lls, u, scene_data);
        }

        LightLiSample Light::Li(LightLiSample lls, const SceneData *data) const {
            LUMINOUS_VAR_PTR_DISPATCH(Li, lls, data);
        }

        LightLiSample Light::sample_Li(float2 u, LightLiSample lls, uint64_t traversable_handle,
                                                       const SceneData *scene_data) const {
            lls.lec = sample(&lls, u, scene_data);
            Ray ray = lls.lsc.spawn_ray_to(lls.lec);
            bool occluded = intersect_any(traversable_handle, ray);
            lls = Li(lls, scene_data);
            lls.update_Li();
            return select(occluded, LightLiSample(), lls);
        }

        bool Light::is_delta() const {
            LUMINOUS_VAR_PTR_DISPATCH(is_delta);
        }

        bool Light::is_infinite() const {
            LUMINOUS_VAR_PTR_DISPATCH(is_infinite);
        }

        Spectrum Light::on_miss(float3 dir, const SceneData *data) const {
            LUMINOUS_VAR_PTR_DISPATCH(on_miss, dir, data);
        }

        float Light::PDF_Li(const LightSampleContext &ctx, const LightEvalContext &p_light,
                            float3 wi, const SceneData *data) const {
            LUMINOUS_VAR_PTR_DISPATCH(PDF_Li, ctx, p_light, wi, data);
        }

        Spectrum Light::power() const {
            LUMINOUS_VAR_PTR_DISPATCH(power);
        }

        void Light::print() const {
            LUMINOUS_VAR_PTR_DISPATCH(print);
        }

        CPU_ONLY(LM_NODISCARD std::string Light::to_string() const {
            LUMINOUS_VAR_PTR_DISPATCH(to_string);
        })

        CPU_ONLY(std::pair<Light, std::vector<size_t>> Light::create(const LightConfig &config) {
            return detail::create_ptr<Light>(config);
        })

    } // luminous::render
} // luminous