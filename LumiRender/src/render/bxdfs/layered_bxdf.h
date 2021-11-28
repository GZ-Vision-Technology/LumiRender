//
// Created by Zero on 28/11/2021.
//


#pragma once

#include "base_libs/optics/rgb.h"

namespace luminous {
    inline namespace render {
        template<typename TopBxDF, typename BottomBxDF, bool twoSided>
        class LayeredBxDF {
        private:
            TopBxDF _top{};
            BottomBxDF _bottom{};
            Spectrum _albedo{};
            float _thickness{};
            float _g{};
            int _max_depth{};
            int _sample_num{};
        public:
            LM_XPU LayeredBxDF() = default;

            LM_XPU LayeredBxDF(TopBxDF top, BottomBxDF bottom, float thickness,
                               const Spectrum &albedo, float g, int maxDepth, int nSamples)
                    : _top(top),
                      _bottom(bottom),
                      _thickness(std::max(thickness, std::numeric_limits<float>::min())),
                      _g(g),
                      _albedo(albedo),
                      _max_depth(maxDepth),
                      _sample_num(nSamples) {}

            LM_XPU void regularize() {
                _top.regularize();
                _bottom.regularize();
            }
        };
    }
}