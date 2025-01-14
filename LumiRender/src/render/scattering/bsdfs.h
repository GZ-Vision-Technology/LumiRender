//
// Created by Zero on 16/12/2021.
//


#pragma once

#include "bsdf_ty.h"
#include "microfacet.h"
#include "bsdf_data.h"
#include "lambert_scatter.h"
#include "microfacet_scatter.h"
#include "specular_scatter.h"
#include "disney_bsdf.h"

namespace luminous {
    inline namespace render {

        using DiffuseBSDF = BSDF_Ty<BSDFHelper, true, LambertReflection>;

        ND_XPU_INLINE DiffuseBSDF create_diffuse_bsdf(float3 color) {
            return DiffuseBSDF(BSDFHelper::create_diffuse_data(color), LambertReflection{color});
        }

        using OrenNayarBSDF = BSDF_Ty<BSDFHelper, true, OrenNayar>;

        ND_XPU_INLINE OrenNayarBSDF create_oren_nayar_bsdf(float3 color, float sigma) {
            return OrenNayarBSDF(BSDFHelper::create_oren_nayar_data(color, sigma), OrenNayar{color});
        }

        using MirrorBSDF = BSDF_Ty<BSDFHelper, true, SpecularReflection>;

        ND_XPU_INLINE MirrorBSDF create_mirror_bsdf(float3 color) {
            return MirrorBSDF(BSDFHelper::create_mirror_data(color),
                              SpecularReflection{color});
        }

        using GlassBSDF = FresnelBSDF<BSDFHelper, SpecularReflection, SpecularTransmission, true>;

        ND_XPU_INLINE GlassBSDF create_glass_bsdf(float3 color, float eta,
                                                  bool valid_refl = true, bool valid_trans = true) {
            return GlassBSDF(BSDFHelper::create_glass_data(color, eta),
                             SpecularReflection{color}, SpecularTransmission{color});
        }

        using RoughGlassBSDF = FresnelBSDF<BSDFHelper, MicrofacetReflection, MicrofacetTransmission>;

        ND_XPU_INLINE RoughGlassBSDF
        create_rough_glass_bsdf(float3 color, float eta, float alpha_x, float alpha_y,
                                bool valid_refl = true, bool valid_trans = true) {
            auto param = BSDFHelper::create_glass_data(color, eta);
            return RoughGlassBSDF(param,
                                  MicrofacetReflection{color, alpha_x, alpha_y, GGX},
                                  MicrofacetTransmission{color, alpha_x, alpha_y, GGX});
        }

        using FakeMetalBSDF = BSDF_Ty<BSDFHelper, true, MicrofacetReflection>;

        ND_XPU_INLINE FakeMetalBSDF create_fake_metal_bsdf(float3 color, float alpha_x, float alpha_y) {
            auto param = BSDFHelper::create_fake_metal_data(color);
            return FakeMetalBSDF(param, MicrofacetReflection{color, alpha_x, alpha_y, GGX});
        }

        using MetalBSDF = BSDF_Ty<BSDFHelper, true, MicrofacetReflection>;

        ND_XPU_INLINE MetalBSDF create_metal_bsdf(float3 eta, float3 k, float alpha_x, float alpha_y) {
            BSDFHelper data = BSDFHelper::create_metal_data(make_float4(eta, 1.f), k);
            return MetalBSDF(data, MicrofacetReflection{Spectrum{1.f}, alpha_x, alpha_y, GGX});
        }

        using DisneyBSDF = BSDF_Ty<BSDFHelper, false,
                disney::DiffuseLobes,
//        disney::Diffuse, disney::FakeSS,
//                disney::Retro, disney::Sheen,
                disney::Clearcoat,
                MicrofacetReflection, MicrofacetTransmission,
                LambertTransmission, SpecularTransmission>;

        using NormalizedFresnelBSDF = BSDF_Ty<BSDFHelper, false, NormalizedFresnelBxDF>;

        using SubstrateBSDF = BSDF_Ty<BSDFHelper, true, MicrofacetFresnel>;

        using NeubeltClothBSDF = BSDF_Ty<BSDFHelper, true, ClothMicrofacetFresnel>;

        ND_XPU_INLINE SubstrateBSDF create_substrate_bsdf(float3 color, float3 spec, float2 alpha, float eta = 1.5f) {
            BSDFHelper data = BSDFHelper::create_substrate_data(1.5);
            return SubstrateBSDF(data, MicrofacetFresnel{Spectrum{color}, Spectrum{spec}, alpha.x, alpha.y, GGX});
        }

        struct DisneyBSDFData {
        public:
            float4 color{make_float4(0.f)};
            float metallic{0.f};
            float eta{1.5};
            float spec_trans{0.f};
            float diff_trans{0.f};
            float spec_tint{0.f};
            float roughness{1.f};
            float sheen_weight{0.f};
            float sheen_tint{0.f};
            float clearcoat{0.f};
            float4 scatter_distance{make_float4(0.f)};
            float clearcoat_roughness{1.f};
            float anisotropic{0.f};
            float flatness{0.f};
            bool thin{false};

            LM_ND_XPU DisneyBSDF create() const;
        };


        class BSDF : public Variant<DiffuseBSDF, MirrorBSDF,
                GlassBSDF, RoughGlassBSDF, DisneyBSDF,
                FakeMetalBSDF, MetalBSDF, SubstrateBSDF> {

        DECLARE_REFLECTION(BSDF, Variant)

        private:
            using Variant::Variant;
        public:
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