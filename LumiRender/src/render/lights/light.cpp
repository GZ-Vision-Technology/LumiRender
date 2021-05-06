//
// Created by Zero on 2021/4/7.
//

#include "light.h"
#include "render/include/creator.h"

namespace luminous {
    inline namespace render {

        LightType Light::type() const {
            LUMINOUS_VAR_DISPATCH(type);
        }

        SurfaceInteraction Light::sample(float2 u, const HitGroupData *hit_group_data) const {
            LUMINOUS_VAR_DISPATCH(sample, u, hit_group_data);
        }

        LightLiSample Light::Li(LightLiSample lls) const {
            LUMINOUS_VAR_DISPATCH(Li, lls);
        }

        lstd::optional<LightLiSample> Light::sample_Li(float2 u, LightLiSample lls, uint64_t traversable_handle,
                                                       const HitGroupData *hit_group_data) const {
            LUMINOUS_VAR_DISPATCH(sample_Li, u, lls, traversable_handle, hit_group_data);
        }

        float3 Light::estimate_direct_lighting(const LightSampleContext &ctx, const BSDF &bsdf, Sampler &sampler,
                                               uint64_t traversable_handle, const HitGroupData *hit_group_data,
                                               float3 *wi) const {
            // todo
            return float3();
        }

        bool Light::is_delta() const {
            LUMINOUS_VAR_DISPATCH(is_delta);
        }

        float Light::PDF_dir(const Interaction &ref_p, const SurfaceInteraction &p_light) const {
            LUMINOUS_VAR_DISPATCH(PDF_dir, ref_p, p_light);
        }

        float3 Light::power() const {
            LUMINOUS_VAR_DISPATCH(power);
        }

        Light Light::create(const LightConfig &config) {
            return detail::create<Light>(config);
        }

    } // luminous::render
} // luminous