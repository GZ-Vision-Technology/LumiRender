//
// Created by Zero on 2021/1/29.
//


#pragma once

#include "graphics/geometry/common.h"
#include "graphics/optics/common.h"
#include "render/include/interaction.h"
#include "render/include/scene_graph.h"

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
            float3 L{};
            float3 wi{};
            float PDF_dir{-1.f};
            Interaction p_light{};
            Interaction p_ref{};
            XPU LightLiSample() = default;

            XPU LightLiSample(const float3 &L, float3 wi,
                              float PDF, const Interaction &i)
                    : L(L), wi(wi), PDF_dir(PDF), p_light(i) {}
        };

        struct LightSampleContext {
            float3 pos;
            float3 ng;
            float3 ns;
            XPU LightSampleContext() = default;

            XPU LightSampleContext(float3 p, float3 ng, float3 ns)
                    : pos(p), ng(ng), ns(ns) {}
        };

        class LightBase {
        protected:
            const LightType _type;
        public:
            XPU LightBase(LightType type) : _type(type) {}

            NDSC_XPU LightType type() const {
                return _type;
            }

            NDSC_XPU bool is_delta() const {
                return _type == LightType::DeltaDirection || _type == LightType::DeltaPosition;
            }

            NDSC std::string _to_string() const {
                return string_printf("type : %d", (int) _type);
            }
        };
    }
}