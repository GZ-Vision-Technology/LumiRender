//
// Created by Zero on 2021/1/31.
//


#pragma once

#include "uniform.h"
#include "power.h"
#include "bvh.h"
#include "base_libs/lstd/lstd.h"
#include "render/bxdfs/bsdf.h"
#include "render/samplers/sampler.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;
        using lstd::optional;

        class LightSampler : BASE_CLASS(Variant<UniformLightSampler>) {
        public:
            using BaseBinder::BaseBinder;

            REFL_CLASS(LightSampler)

            GEN_BASE_NAME(LightSampler)

            GEN_TO_STRING_FUNC

            XPU void set_lights(BufferView<const Light> lights);

            XPU void set_infinite_lights(BufferView<const Light> lights);

            NDSC_XPU BufferView<const Light> lights() const;

            NDSC_XPU const Light &light_at(uint idx) const;

            NDSC_XPU size_t light_num() const;

            XPU void for_each_light(const std::function<void(const Light&, int i)> &func) const;

            XPU void for_each_infinite_light(const std::function<void(const Envmap&, int i)> &func) const;

            NDSC_XPU SampledLight sample(float u) const;

            NDSC_XPU SampledLight sample(const LightSampleContext &ctx, float u) const;

            NDSC_XPU Spectrum estimate_direct_lighting(const SurfaceInteraction &si, Sampler &sampler,
                                                       uint64_t traversable_handle,
                                                       const SceneData *scene_data, NEEData *NEE_data) const;

            NDSC_XPU float PMF(const Light &light) const;

            NDSC_XPU float PMF(const LightSampleContext &ctx, const Light &light) const;

            CPU_ONLY(static LightSampler create(const LightSamplerConfig &config);)
        };

    } // luminous::render
} // luminous