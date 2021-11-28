//
// Created by Zero on 28/11/2021.
//


#pragma once

#include "base_libs/optics/rgb.h"
#include "base.h"

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

            LM_ND_XPU BxDFFlags flags() const {
                BxDFFlags top_flags = _top.flags();
                BxDFFlags bottom_flags = _bottom.flags();
                DCHECK(is_transmissive(top_flags) || is_transmissive(bottom_flags));

                BxDFFlags flags = BxDFFlags::Reflection;
                if (is_specular(top_flags))
                    flags = flags | BxDFFlags::Specular;

                if (is_diffuse(top_flags) || is_diffuse(bottom_flags) || _albedo.is_black()) {
                    flags = flags | BxDFFlags::Diffuse;
                }
                else if (is_glossy(top_flags) || is_glossy(bottom_flags)) {
                    flags = flags | BxDFFlags::Glossy;
                }
                if (is_transmissive(top_flags) && is_transmissive(bottom_flags)) {
                    flags = flags | BxDFFlags::Transmission;
                }
                return flags;
            }

            LM_XPU void regularize() {
                _top.regularize();
                _bottom.regularize();
            }
        };
    }
}