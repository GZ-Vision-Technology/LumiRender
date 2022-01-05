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
#include "disney_bsdf.h"

namespace luminous {
    inline namespace render {

        using DiffuseBSDF = BSDF_Ty<PhysicallyMaterialData, DiffuseReflection>;

        ND_XPU_INLINE DiffuseBSDF create_diffuse_bsdf(float4 color) {
            return DiffuseBSDF(PhysicallyMaterialData::create_diffuse_data(color), DiffuseReflection{color});
        }

        using OrenNayarBSDF = BSDF_Ty<PhysicallyMaterialData, OrenNayar>;

        ND_XPU_INLINE OrenNayarBSDF create_oren_nayar_bsdf(float4 color, float sigma) {
            return OrenNayarBSDF(PhysicallyMaterialData::create_oren_nayar_data(color, sigma), OrenNayar{color});
        }

        using MirrorBSDF = BSDF_Ty<PhysicallyMaterialData, SpecularReflection>;

        ND_XPU_INLINE MirrorBSDF create_mirror_bsdf(float4 color) {
            return MirrorBSDF(PhysicallyMaterialData::create_mirror_data(color),
                              SpecularReflection{color});
        }

        using GlassBSDF = FresnelBSDF<PhysicallyMaterialData, SpecularReflection, SpecularTransmission, true>;

        ND_XPU_INLINE GlassBSDF create_glass_bsdf(float4 color, float eta,
                                                  bool valid_refl = true, bool valid_trans = true) {
            return GlassBSDF(PhysicallyMaterialData::create_glass_data(color, eta, 0, 0),
                             SpecularReflection{color}, SpecularTransmission{color});
        }

        using RoughGlassBSDF = FresnelBSDF<PhysicallyMaterialData, MicrofacetReflection, MicrofacetTransmission>;

        ND_XPU_INLINE RoughGlassBSDF
        create_rough_glass_bsdf(float4 color, float eta, float alpha_x, float alpha_y,
                                bool valid_refl = true, bool valid_trans = true) {
            auto param = PhysicallyMaterialData::create_glass_data(color, eta, alpha_x, alpha_y);
            return RoughGlassBSDF(param,
                                  MicrofacetReflection{color},
                                  MicrofacetTransmission{color});
        }

        using FakeMetalBSDF = BSDF_Ty<PhysicallyMaterialData, MicrofacetReflection>;

        ND_XPU_INLINE FakeMetalBSDF create_fake_metal_bsdf(float4 color, float alpha_x, float alpha_y) {
            auto param = PhysicallyMaterialData::create_fake_metal_data(color, alpha_x, alpha_y);
            return FakeMetalBSDF(param, MicrofacetReflection{color});
        }

        using MetalBSDF = BSDF_Ty<PhysicallyMaterialData, MicrofacetReflection>;

        ND_XPU_INLINE MetalBSDF create_metal_bsdf(float4 eta, float4 k, float alpha_x, float alpha_y) {
            PhysicallyMaterialData data = PhysicallyMaterialData::create_metal_data(eta, k, alpha_x, alpha_y);
            return MetalBSDF(data, MicrofacetReflection{Spectrum{1.f}});
        }

//        using DisneyBSDF = BSDF_Ty<DisneyMaterialData, disney::Diffuse, disney::FakeSS,
//                disney::Retro, disney::Sheen, disney::Clearcoat,
//                disney::MicrofacetReflection, disney::MicrofacetTransmission,
//                disney::DiffuseTransmission, disney::SpecularTransmission>;


        class BSDF : public Variant<DiffuseBSDF, OrenNayarBSDF, MirrorBSDF,
                GlassBSDF, RoughGlassBSDF,
                FakeMetalBSDF, MetalBSDF> {
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