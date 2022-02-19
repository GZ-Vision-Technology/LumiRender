//
// Created by Zero on 2021/4/7.
//


#pragma once

#include "render/samplers/sampler.h"
#include "base_libs/lstd/lstd.h"
#include "light_util.h"
#include "parser/config.h"
#include "render/scattering/bsdf_wrapper.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;

        class PointLight;

        class AreaLight;

        class Envmap;

        class SpotLight;

        class Light : public Variant<AreaLight *, Envmap *> {

            DECLARE_REFLECTION(Light, Variant)

        private:
            using Variant::Variant;
        public:
            LM_ND_XPU LightType type() const;

            CPU_ONLY(LM_NODISCARD std::string to_string() const;)

            LM_ND_XPU bool is_delta() const;

            LM_ND_XPU bool is_infinite() const;

            LM_ND_XPU Spectrum on_miss(float3 dir, const SceneData *data) const;

            LM_ND_XPU LightEvalContext sample(LightLiSample *lls, float2 u, const SceneData *scene_data) const;

            LM_ND_XPU LightLiSample Li(LightLiSample lls, const SceneData *data) const;

            LM_ND_XPU LightLiSample sample_Li(float2 u, LightLiSample lls, uint64_t traversable_handle,
                                                              const SceneData *scene_data) const;

            LM_ND_XPU Spectrum MIS_sample_light(const SurfaceInteraction &si, const BSDFWrapper &bsdf,
                                                Sampler &sampler, uint64_t traversable_handle,
                                                const SceneData *scene_data) const;

            LM_ND_XPU Spectrum MIS_sample_BSDF(const SurfaceInteraction &si, const BSDFWrapper &bsdf,
                                               Sampler &sampler, uint64_t traversable_handle,
                                               PathVertex *vertex, const SceneData *data) const;

            LM_ND_XPU Spectrum estimate_direct_lighting(const SurfaceInteraction &si,
                                                        Sampler &sampler, uint64_t traversable_handle,
                                                        const SceneData *scene_data, PathVertex *vertex) const;

            LM_ND_XPU float PDF_Li(const LightSampleContext &ctx, const LightEvalContext &p_light,
                                   float3 wi, const SceneData *data) const;

            LM_ND_XPU Spectrum power() const;

            LM_XPU void print() const;

            CPU_ONLY(static std::pair<Light, std::vector<size_t>> create(const LightConfig &config);)
        };
    } // luminous::render
} // luminous