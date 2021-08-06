//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "base_libs/geometry/common.h"
#include "base_libs/optics/common.h"
#include "render/include/interaction.h"
#include "render/include/config.h"
#include "render/include/shader_data.h"

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

        struct LightLiSample {
            Spectrum L{};
            float3 wi{};
            float PDF_dir{-1.f};
            SurfaceInteraction p_light{};
            Interaction p_ref{};
            XPU LightLiSample() = default;

            XPU LightLiSample(const float3 &L, float3 wi,
                              float PDF, const SurfaceInteraction &si)
                    : L(L), wi(wi), PDF_dir(PDF), p_light(si) {}

            NDSC_XPU bool has_contribution() const {
                return nonzero(L) && PDF_dir != 0;
            }
        };

        struct LightSampleContext {
            float3 pos;
            float3 ng;
            float3 ns;
            XPU LightSampleContext() = default;

            XPU LightSampleContext(const SurfaceInteraction &si)
                    : pos(si.pos), ng(si.g_uvn.normal), ns(si.s_uvn.normal) {}

            XPU LightSampleContext(float3 p, float3 ng, float3 ns)
                    : pos(p), ng(ng), ns(ns) {}
        };
        class Scene;
        class MissData;
        class LightBase {
        protected:
            const LightType _type;
            const float3 _miss_color;
        public:
            XPU LightBase(LightType type, float3 miss_color = make_float3(0.f))
                : _type(type), _miss_color(miss_color) {}

            NDSC_XPU LightType type() const {
                return _type;
            }

            NDSC_XPU Spectrum on_miss(Ray ray, const SceneData * data) const {
                return {_miss_color};
            }

            NDSC_XPU bool is_infinite() const {
                return _type == LightType::Infinite;
            }

            NDSC_XPU bool is_delta() const {
                return _type == LightType::DeltaDirection || _type == LightType::DeltaPosition;
            }

            GEN_STRING_FUNC({
                return string_printf("type : %d", (int) _type);
            })
        };
    }
}