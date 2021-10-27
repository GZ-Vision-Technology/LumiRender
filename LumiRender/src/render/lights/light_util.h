//
// Created by Zero on 25/09/2021.
//


#pragma once

#include <utility>

#include "base_libs/geometry/common.h"
#include "base_libs/optics/common.h"
#include "render/include/interaction.h"

namespace luminous {
    inline namespace render {
        // LightType Definition
        enum class LightType {
            DeltaPosition,
            DeltaDirection,
            Area,
            Infinite
        };

        enum class LightSamplingMode {
            WithMIS,
            WithoutMIS
        };

        struct LightSampleContext {
            float3 pos;
            float3 ng;
            float3 ns;
            LM_XPU LightSampleContext() = default;

            LM_XPU explicit LightSampleContext(const SurfaceInteraction &si)
                    : pos(si.pos), ng(si.g_uvn.normal), ns(si.s_uvn.normal) {}

            LM_XPU explicit LightSampleContext(const Interaction &it)
                    : pos(it.pos), ng(it.g_uvn.normal), ns(it.g_uvn.normal) {}

            LM_XPU LightSampleContext(float3 p, float3 ng, float3 ns)
                    : pos(p), ng(ng), ns(ns) {}

            ND_XPU_INLINE Ray spawn_ray(float3 dir) const {
                return Ray::spawn_ray(pos, ng, dir);
            }

            ND_XPU_INLINE Ray spawn_ray_to(float3 p) const {
                return Ray::spawn_ray_to(pos, ng, p);
            }

            ND_XPU_INLINE Ray spawn_ray_to(const LightSampleContext &it) const {
                return Ray::spawn_ray_to(pos, ng, it.pos, it.ng);
            }
        };

        struct LightLiSample {
            Spectrum L{};
            float3 wi{};
            float PDF_dir{-1.f};
            SurfaceInteraction p_light{};
            LightSampleContext ctx{};
            LM_XPU LightLiSample() = default;

            LM_XPU LightLiSample(const float3 &L, float3 wi,
                                 float PDF, SurfaceInteraction si)
                    : L(L), wi(wi), PDF_dir(PDF), p_light(std::move(si)) {}

            LM_ND_XPU bool has_contribution() const {
                return nonzero(L) && PDF_dir != 0;
            }
        };
    }
}