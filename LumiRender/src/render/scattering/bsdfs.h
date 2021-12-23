//
// Created by Zero on 16/12/2021.
//


#pragma once

#include "bsdf_ty.h"
#include "microfacet.h"
#include "bsdf_data.h"
#include "fresnel.h"
#include "diffuse_scatter.h"
#include "microfacet_scatter.h"
#include "specular_scatter.h"

namespace luminous {
    inline namespace render {

        using DiffuseBSDF = BSDF_Ty<BSDFCommonData, FresnelNoOp, DiffuseReflection>;

        ND_XPU_INLINE DiffuseBSDF create_diffuse_bsdf(float4 color) {
            BSDFCommonData data{color};
            return DiffuseBSDF(data, FresnelNoOp{}, Microfacet{}, DiffuseReflection{});
        }

        using OrenNayarBSDF = BSDF_Ty<OrenNayarData, FresnelNoOp, OrenNayar>;

        ND_XPU_INLINE OrenNayarBSDF create_oren_nayar_bsdf(float4 color, float sigma) {
            OrenNayarData data{color, sigma};
            return OrenNayarBSDF(data, FresnelNoOp{}, Microfacet{}, OrenNayar{});
        }

        using MirrorBSDF = BSDF_Ty<BSDFCommonData, FresnelNoOp, SpecularReflection>;

        ND_XPU_INLINE MirrorBSDF create_mirror_bsdf(float4 color) {
            BSDFCommonData data{color};
            return MirrorBSDF(data, FresnelNoOp{}, Microfacet{}, SpecularReflection{});
        }

        using GlassBSDF = BSDF_Ty<BSDFCommonData, FresnelDielectric, SpecularFresnel>;

        ND_XPU_INLINE GlassBSDF create_glass_bsdf(float4 color, float eta) {
            BSDFCommonData data{color};
            return GlassBSDF(data, FresnelDielectric{eta}, Microfacet{}, SpecularFresnel{});
        }

        using GlassBSDFForTest = BSDF_Ty<BSDFCommonData, FresnelDielectric,
        SpecularReflection, SpecularTransmission>;

        ND_XPU_INLINE GlassBSDFForTest create_glass_bsdf_test(float4 color, float eta,
                                                              bool valid_refl = true, bool valid_trans = true) {
            BSDFCommonData data{color};
            return GlassBSDFForTest(data, FresnelDielectric{eta}, Microfacet{},
                                    SpecularReflection{valid_refl}, SpecularTransmission{valid_trans});
        }

        using RoughGlassBSDF = BSDF_Ty<BSDFCommonData, FresnelDielectric, MicrofacetFresnel>;

        ND_XPU_INLINE RoughGlassBSDF create_rough_glass_bsdf(float4 color, float eta, float alpha_x, float alpha_y) {
            BSDFCommonData data{color};
            return RoughGlassBSDF(data, FresnelDielectric{eta}, Microfacet{alpha_x, alpha_y,GGX}, MicrofacetFresnel{});
        }

        using RoughGlassBSDFForTest = BSDF_Ty<BSDFCommonData, FresnelDielectric, MicrofacetReflection, MicrofacetTransmission>;

        ND_XPU_INLINE RoughGlassBSDFForTest
        create_rough_glass_bsdf_test(float4 color, float eta, float alpha_x, float alpha_y,
                                     bool valid_refl = true, bool valid_trans = true) {
            BSDFCommonData data{color};
            return RoughGlassBSDFForTest(data, FresnelDielectric{eta}, Microfacet{alpha_x, alpha_y},
                                         MicrofacetReflection{valid_refl}, MicrofacetTransmission{valid_trans});
        }

        class BSDF : public Variant<DiffuseBSDF, OrenNayarBSDF, MirrorBSDF,
                GlassBSDF, RoughGlassBSDFForTest, GlassBSDFForTest, RoughGlassBSDF> {
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