//
// Created by Zero on 2021/1/31.
//


#pragma once

#include "uniform.h"
#include "power.h"
#include "bvh.h"
#include "graphics/lstd/lstd.h"
#include "render/bxdfs/bsdf.h"
#include "render/samplers/sampler.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;
        using lstd::optional;

        class LightSampler : public Variant<UniformLightSampler> {
        public:
            using Variant::Variant;

            GEN_BASE_NAME(LightSampler)

            GEN_NAME_AND_TO_STRING_FUNC

            XPU void set_lights(BufferView<const Light> lights);

            NDSC_XPU BufferView<const Light> lights() const;

            NDSC_XPU size_t light_num();

            NDSC_XPU lstd::optional<SampledLight> sample(float u) const;

            NDSC_XPU lstd::optional<SampledLight> sample(const LightSampleContext &ctx, float u) const;

            NDSC_XPU Spectrum estimate_direct_lighting(const SurfaceInteraction &si, const BSDF &bsdf,
                                                     Sampler &sampler, uint64_t traversable_handle,
                                                     const HitGroupData *hit_group_data, float3 *wi,
                                                     Spectrum *bsdf_v) const;

            NDSC_XPU float PMF(const Light &light) const;

            NDSC_XPU float PMF(const LightSampleContext &ctx, const Light &light) const;

            static LightSampler create(const LightSamplerConfig &config);
        };

    } // luminous::render
} // luminous