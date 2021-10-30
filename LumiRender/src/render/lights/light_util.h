//
// Created by Zero on 25/09/2021.
//


#pragma once

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

        struct SurfacePoint {
            float3 pos;
            float3 ng;

            LM_XPU SurfacePoint() = default;

            LM_XPU SurfacePoint(float3 p, float3 n)
                    : pos(p), ng(n) {}

            LM_XPU explicit SurfacePoint(const Interaction &it)
                    : pos(it.pos), ng(it.g_uvn.normal) {}

            LM_XPU explicit SurfacePoint(const SurfaceInteraction &it)
                    : pos(it.pos), ng(it.g_uvn.normal) {}

            ND_XPU_INLINE Ray spawn_ray(float3 dir) const {
                return Ray::spawn_ray(pos, ng, dir);
            }

            ND_XPU_INLINE Ray spawn_ray_to(float3 p) const {
                return Ray::spawn_ray_to(pos, ng, p);
            }

            ND_XPU_INLINE Ray spawn_ray_to(const SurfacePoint &lsc) const {
                return Ray::spawn_ray_to(pos, ng, lsc.pos, lsc.ng);
            }
        };

        struct LightSampleContext : public SurfacePoint {
            float3 ns;
            LM_XPU LightSampleContext() = default;

            LM_XPU explicit LightSampleContext(const Interaction &it)
                    : SurfacePoint(it), ns(it.g_uvn.normal) {}

            LM_XPU LightSampleContext(float3 p, float3 ng, float3 ns)
                    : SurfacePoint{p, ng}, ns(ns) {}


        };

        /**
         * A point on light
         * used to eval light PDF or lighting to LightSampleContext
         */
        struct LightEvalContext : public SurfacePoint {
        public:
            float2 uv{};
            float PDF_pos{};
        public:
            LM_XPU LightEvalContext() = default;

            LM_XPU LightEvalContext(float3 p, float3 ng, float2 uv, float PDF_pos)
                    : SurfacePoint{p, ng}, uv(uv), PDF_pos(PDF_pos) {}

            LM_XPU explicit LightEvalContext(const SurfaceInteraction &si)
                    : SurfacePoint(si), uv(si.uv), PDF_pos(si.PDF_pos) {}
        };

        struct LightLiSample {
            Spectrum L{};
            float3 wi{};
            float PDF_dir{-1.f};
            LightEvalContext p_light{};
            LightSampleContext lsc{};
            LM_XPU LightLiSample() = default;

            LM_XPU LightLiSample(const float3 &L, float3 wi,
                                 float PDF, LightEvalContext lec)
                    : L(L), wi(wi), PDF_dir(PDF), p_light(lec) {}

            LM_ND_XPU bool has_contribution() const {
                return nonzero(L) && PDF_dir != 0;
            }
        };
    }
}