//
// Created by Zero on 16/12/2021.
//


#pragma once

#include "bsdf_ty.h"
#include "microfacet.h"
#include "bsdf_data.h"
#include "diffuse_scatter.h"
#include "microfacet_scatter.h"
#include "specular_scatter.h"

namespace luminous {
    inline namespace render {

        using DiffuseBSDF = BSDF_Ty<DiffuseReflection>;

        ND_XPU_INLINE DiffuseBSDF create_diffuse_bsdf(float4 color) {
            return DiffuseBSDF(BSDFParam::create_diffuse_data(color), MicrofacetDistrib{}, DiffuseReflection{});
        }

        using OrenNayarBSDF = BSDF_Ty<OrenNayar>;

        ND_XPU_INLINE OrenNayarBSDF create_oren_nayar_bsdf(float4 color, float sigma) {
            return OrenNayarBSDF(BSDFParam::create_oren_nayar_data(color, sigma), MicrofacetDistrib{}, OrenNayar{});
        }

        using MirrorBSDF = BSDF_Ty<SpecularReflection>;

        ND_XPU_INLINE MirrorBSDF create_mirror_bsdf(float4 color) {
            return MirrorBSDF(BSDFParam::create_mirror_data(color),
                              MicrofacetDistrib{}, SpecularReflection{});
        }

        using GlassBSDFSingle = BSDF_Ty<SpecularFresnel>;

        ND_XPU_INLINE GlassBSDFSingle create_glass_bsdf_single(float4 color, float eta) {
            return GlassBSDFSingle(BSDFParam::create_glass_data(color, eta), MicrofacetDistrib{},
                                   SpecularFresnel{});
        }

        using GlassBSDF = FresnelBSDF<SpecularReflection, SpecularTransmission, true>;

        ND_XPU_INLINE GlassBSDF create_glass_bsdf(float4 color, float eta,
                                                  bool valid_refl = true, bool valid_trans = true) {
            return GlassBSDF(BSDFParam::create_glass_data(color, eta), MicrofacetDistrib{},
                             SpecularReflection{valid_refl}, SpecularTransmission{valid_trans});
        }

        using RoughGlassBSDFSingle = BSDF_Ty<MicrofacetFresnel>;

        ND_XPU_INLINE RoughGlassBSDFSingle
        create_rough_glass_bsdf_single(float4 color, float eta, float alpha_x, float alpha_y) {
            return RoughGlassBSDFSingle(BSDFParam::create_glass_data(color, eta),
                                        MicrofacetDistrib{alpha_x, alpha_y, GGX}, MicrofacetFresnel{});
        }

        using RoughGlassBSDF = FresnelBSDF<MicrofacetReflection, MicrofacetTransmission>;

        ND_XPU_INLINE RoughGlassBSDF
        create_rough_glass_bsdf(float4 color, float eta, float alpha_x, float alpha_y,
                                bool valid_refl = true, bool valid_trans = true) {
            return RoughGlassBSDF(BSDFParam::create_glass_data(color, eta),
                                  MicrofacetDistrib{alpha_x, alpha_y},
                                  MicrofacetReflection{valid_refl},
                                  MicrofacetTransmission{valid_trans});
        }

        using FakeMetalBSDF = BSDF_Ty<MicrofacetReflection>;

        ND_XPU_INLINE FakeMetalBSDF create_fake_metal_bsdf(float4 color, float roughness_x, float roughness_y) {
            return FakeMetalBSDF(BSDFParam::create_fake_metal_data(color),
                                 MicrofacetDistrib{roughness_x, roughness_y, GGX},
                                 MicrofacetReflection{});
        }

        using MetalBSDF = BSDF_Ty<MicrofacetReflection>;
        ND_XPU_INLINE MetalBSDF create_metal_bsdf(float4 eta, float4 k, float roughness_x, float roughness_y) {
            BSDFParam data = BSDFParam::create_metal_data(eta, k);
            return MetalBSDF(data, MicrofacetDistrib{roughness_x, roughness_y, GGX}, MicrofacetReflection{});
        }

        class BSDF : public Variant<DiffuseBSDF, OrenNayarBSDF, MirrorBSDF,
                GlassBSDF, RoughGlassBSDFSingle, GlassBSDFSingle, RoughGlassBSDF,
                FakeMetalBSDF,MetalBSDF> {
        private:
            using Variant::Variant;
        public:
            GEN_BASE_NAME(BSDF)

            LM_ND_XPU Spectrum color() const {
                LUMINOUS_VAR_DISPATCH(color);
            }

            LM_ND_XPU Spectrum eval(float3 wo, float3 wi, BxDFFlags flags = BxDFFlags::All,
                                    TransportMode mode = TransportMode::Radiance) const {
                LUMINOUS_VAR_DISPATCH(eval, wo, wi, flags, mode);
            }

            LM_ND_XPU float PDF(float3 wo, float3 wi,
                                BxDFFlags flags = BxDFFlags::All,
                                TransportMode mode = TransportMode::Radiance) const {
                LUMINOUS_VAR_DISPATCH(PDF, wo, wi, flags, mode);
            }

            LM_ND_XPU BSDFSample sample_f(float3 wo, float uc, float2 u,
                                          BxDFFlags flags = BxDFFlags::All,
                                          TransportMode mode = TransportMode::Radiance) const {
                LUMINOUS_VAR_DISPATCH(sample_f, wo, uc, u, flags, mode);
            }

            LM_ND_XPU int match_num(BxDFFlags bxdf_flags) const {
                LUMINOUS_VAR_DISPATCH(match_num, bxdf_flags);
            }

            LM_ND_XPU BxDFFlags flags() const {
                LUMINOUS_VAR_DISPATCH(flags);
            }
        };
    }
}