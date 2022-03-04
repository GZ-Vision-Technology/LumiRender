//
// Created by Zero on 2021/1/31.
//


#pragma once

#include "base_libs/lstd/lstd.h"
#include "render/scattering/bsdf_wrapper.h"
#include "render/samplers/sampler.h"


namespace luminous {
    inline namespace render {

        using lstd::Variant;

        class UniformLightSampler;

        class Light;

        struct SampledLight;

        struct PathVertex;

        struct LightSampleContext;

        struct SceneData;

        class SurfaceInteraction;

        class LightSampler : public Variant<UniformLightSampler *> {

        DECLARE_REFLECTION(LightSampler, Variant)

        public:
            using Variant::Variant;

            CPU_ONLY(LM_NODISCARD std::string to_string() const;)

            LM_XPU void set_lights(BufferView<const Light> lights);

            LM_XPU void set_infinite_lights(BufferView<const Light> lights);

            LM_ND_XPU BufferView<const Light> lights() const;

            LM_ND_XPU BufferView<const Light> infinite_lights() const;

            LM_ND_XPU const Light &light_at(uint idx) const;

            LM_ND_XPU const Light &infinite_light_at(uint idx) const;

            LM_ND_XPU size_t light_num() const;

            LM_ND_XPU size_t infinite_light_num() const;

            LM_ND_XPU SampledLight sample(float u) const;

            LM_ND_XPU SampledLight sample(const LightSampleContext &ctx, float u) const;

            LM_ND_XPU Spectrum on_miss(float3 dir, const SceneData *scene_data, Spectrum throughput) const;

            LM_ND_XPU Spectrum MIS_sample_light(const SurfaceInteraction &si, const BSDFWrapper &bsdf, Sampler &sampler,
                                                uint64_t traversable_handle, const SceneData *scene_data) const;

            LM_ND_XPU Spectrum MIS_sample_BSDF(const SurfaceInteraction &si, const BSDFWrapper &bsdf,
                                               Sampler &sampler, uint64_t traversable_handle, PathVertex *vertex,
                                               const SceneData *data) const;

            LM_ND_XPU Spectrum estimate_direct_lighting(const SurfaceInteraction &si, Sampler &sampler,
                                              uint64_t traversable_handle,
                                              const SceneData *scene_data, PathVertex *vertex) const;

            LM_ND_XPU float PMF(const Light &light) const;

            LM_ND_XPU float PMF(const LightSampleContext &ctx, const Light &light) const;

            CPU_ONLY(static std::pair<LightSampler, std::vector<size_t>> create(const LightSamplerConfig &config);)
        };

    } // luminous::render
} // luminous