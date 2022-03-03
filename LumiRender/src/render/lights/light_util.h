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
        enum class LightType : uint8_t {
            DeltaPosition,
            DeltaDirection,
            Area,
            Infinite
        };

        enum class LightSamplingMode : uint8_t {
            WithMIS,
            WithoutMIS
        };

        struct LightSampleContext : public SurfacePoint {
            float3 ns;
            LM_XPU LightSampleContext() = default;

            LM_XPU explicit LightSampleContext(const Interaction &it)
                    : SurfacePoint(it), ns(it.g_uvn.normal()) {}

            LM_XPU explicit LightSampleContext(const SurfaceInteraction &it)
                    : SurfacePoint(it), ns(it.s_uvn.normal()) {}

            LM_XPU LightSampleContext(float3 p, float3 ng, float3 ns)
                    : SurfacePoint{p, ng}, ns(ns) {}


        };

        struct LightLiSample {
            Spectrum L{};
            float3 wi{};
            float PDF_dir{-1.f};
            LightEvalContext lec{};
            LightSampleContext lsc{};
            LM_XPU LightLiSample() = default;

            LM_XPU LightLiSample(LightSampleContext lsc,
                                 LightEvalContext lec)
                    : lsc(lsc), lec(lec) {
                wi = normalize(lec.pos - lsc.pos);
            }

            LM_XPU explicit LightLiSample(LightSampleContext lsc, float3 wi = make_float3(0.f))
                    : lsc(lsc), wi(wi) {}

            LM_XPU LightLiSample(const float3 &L, float3 wi,
                                 float PDF, LightEvalContext lec)
                    : L(L), wi(wi), PDF_dir(PDF), lec(lec) {}

            ND_XPU_INLINE bool valid() const {
                return PDF_dir > 0.f;
            }

            LM_XPU void compute_PDF_dir() {
                float3 wi_un = lec.pos - lsc.pos;
                float PDF = luminous::PDF_dir(lec.PDF_pos, lsc.ng, -wi_un);
                PDF_dir = select(is_inf(PDF), 0, PDF);
            }

            LM_XPU void set_sample_result(float PDF, LightEvalContext ctx, float3 w) {
                PDF_dir = PDF;
                lec = ctx;CHECK_UNIT_VEC(w)
                wi = w;
            }

            LM_XPU void set_Li(Spectrum Li) {
                L = Li;
            }

            LM_XPU void update_Li() {
                float factor = PDF_dir == 0 ? 0 : 1;
                L *= factor;
            }

            LM_ND_XPU bool has_contribution() const {
                return nonzero(L) && PDF_dir != 0;
            }
        };
    }
}