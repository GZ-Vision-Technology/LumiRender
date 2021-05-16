//
// Created by Zero on 2021/4/7.
//


#pragma once

#include "light_base.h"
#include "point_light.h"
#include "spot_light.h"
#include "area_light.h"
#include "render/samplers/sampler.h"
#include "graphics/lstd/lstd.h"
#include "render/include/config.h"
#include "render/bxdfs/bsdf.h"

namespace luminous {
    inline namespace render {

        using lstd::Variant;

        class Light : public Variant<PointLight, AreaLight> {
        private:
            using Variant::Variant;
        public:
            GEN_BASE_NAME(Light)

            NDSC_XPU LightType type() const;

            GEN_NAME_AND_TO_STRING_FUNC

            NDSC_XPU bool is_delta() const;

            NDSC_XPU SurfaceInteraction sample(float2 u, const HitGroupData *hit_group_data) const;

            NDSC_XPU LightLiSample Li(LightLiSample lls) const;

            NDSC_XPU lstd::optional<LightLiSample> sample_Li(float2 u, LightLiSample lls, uint64_t traversable_handle,
                                                             const HitGroupData *hit_group_data) const;

            NDSC_XPU Spectrum MIS_sample_light(const SurfaceInteraction &si,
                                               Sampler &sampler, uint64_t traversable_handle,
                                               const HitGroupData *hit_group_data) const;

            NDSC_XPU Spectrum MIS_sample_BSDF(const SurfaceInteraction &si,
                                              Sampler &sampler, uint64_t traversable_handle,
                                              NEEData *NEE_data) const;

            NDSC_XPU Spectrum estimate_direct_lighting(const SurfaceInteraction &si,
                                                       Sampler &sampler, uint64_t traversable_handle,
                                                       const HitGroupData *hit_group_data, NEEData *NEE_data) const;

            NDSC_XPU float PDF_dir(const Interaction &ref_p, const SurfaceInteraction &p_light) const;

            NDSC_XPU Spectrum power() const;

            XPU void print() const;

            static Light create(const LightConfig &config);
        };
    } // luminous::render
} // luminous