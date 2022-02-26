//
// Created by Zero on 27/12/2021.
//


#pragma once

#include "base.h"
#include "bsdf_data.h"
#include "bsdf_ty.h"
#include "lambert_scatter.h"
#include "microfacet_scatter.h"
#include "specular_scatter.h"

namespace luminous {
    inline namespace render {
        namespace disney {
            class Diffuse : public ColoredBxDF<Diffuse> {
            public:
                using ColoredBxDF::ColoredBxDF;

                LM_XPU explicit Diffuse(Spectrum color) : ColoredBxDF(color, DiffRefl) {}

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const;
            };

            class FakeSS : public ColoredBxDF<FakeSS> {
            public:
                using ColoredBxDF::ColoredBxDF;
            public:
                LM_XPU explicit FakeSS(Spectrum color) : ColoredBxDF(color, DiffRefl) {}

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const;

            };

            class Retro : public ColoredBxDF<Retro> {
            public:
                using ColoredBxDF::ColoredBxDF;
            public:
                LM_XPU explicit Retro(Spectrum color) : ColoredBxDF(color, DiffRefl) {}

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const;

                ND_XPU_INLINE float weight(BSDFHelper helper, Spectrum Fr) const {
                    return luminance(spectrum() * Fr);
                }

                LM_ND_XPU float safe_PDF(float3 wo, float3 wi,
                                         BSDFHelper helper,
                                         TransportMode mode = TransportMode::Radiance) const;

                LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                              TransportMode mode = TransportMode::Radiance) const;
            };

            class Sheen : public ColoredBxDF<Sheen> {
            public:
                using ColoredBxDF::ColoredBxDF;
            public:
                LM_XPU explicit Sheen(Spectrum color) : ColoredBxDF(color, DiffRefl) {}

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const;

            };

            class Clearcoat : public BxDF<Clearcoat> {
            private:
                float _weight{};
            public:
                using BxDF::BxDF;
            public:
                LM_XPU explicit Clearcoat(float weight)
                        : BxDF(weight > 0 ? BxDFFlags::GlossyRefl : BxDFFlags::Unset),
                          _weight(weight) {}

                LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BSDFHelper helper,
                                        TransportMode mode = TransportMode::Radiance) const;

                ND_XPU_INLINE Spectrum color(BSDFHelper helper) const {
                    return Spectrum(_weight);
                }

                /**
                 * must be reflection
                 */
                LM_ND_XPU float PDF(float3 wo, float3 wi,
                                    BSDFHelper helper,
                                    TransportMode mode = TransportMode::Radiance) const;

                LM_ND_XPU float safe_PDF(float3 wo, float3 wi,
                                         BSDFHelper helper,
                                         TransportMode mode = TransportMode::Radiance) const;

                LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u, BSDFHelper helper,
                                              TransportMode mode = TransportMode::Radiance) const;

                ND_XPU_INLINE float weight(BSDFHelper helper, Spectrum Fr) const {
                    return _weight;
                }
            };
        }
    }
}