//
// Created by Zero on 2020/12/31.
//

#pragma once

#include "graphics/math/common.h"
#include "graphics/lstd/lstd.h"
#include "scene_graph.h"

namespace luminous {
    inline namespace render {
        struct FilterSample {
            float2 p;
            float weight;
        };

        struct FilterSampler {
        private:
            Box2f _domain;
            PiecewiseConstant2D _distrib;
        public:

            NDSC_XPU integral() const {
                return _distrib.Integral();
            }

            NDSC std::string to_string() const {
                // todo implement
                return string_printf("to do implement");
            }
        }

        struct FilterBase {
        protected:
            const float2 _radius;
        public:
            FilterBase(const float2 r) : _radius(r){}
            NDSC_XPU float2 radius() {
                return _radius;
            }
        }

        struct FilterSample;
        class BoxFilter;
        class GaussianFilter;
        class TriangleFilter;

        using lstd::Variant;

        class FilterHandle : public Variant<BoxFilter *, GaussianFilter *, TriangleFilter *> {
            using Variant::Variant;
        public:
            NDSC std::string to_string() const;

            NDSC_XPU float2 radius() const;

            NDSC_XPU float integral() const;

            NDSC_XPU float evaluate(float2 p) const;

            NDSC_XPU FilterSample sample(float2 u) const;

            static FilterHandle create(const FilterConfig &config);
        }
    }
}