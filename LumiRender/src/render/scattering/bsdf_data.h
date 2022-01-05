//
// Created by Zero on 16/12/2021.
//


#pragma once

#include "base_libs/optics/rgb.h"
#include "base_libs/optics/optics.h"

namespace luminous {
    inline namespace render {

        enum FresnelType : uint8_t {
            NoOp = 0,
            Dielectric,
            DisneyFr,
            Conductor
        };

        class DisneyMaterialData;

        class PhysicallyMaterialData;

        struct BSDFHelper {
        private:
            // eta for metal
            float4 _params{};

            // k for metal
            float val0{};
            float val1{};
            float val2{};
            FresnelType _fresnel_type{NoOp};
        public:
            LM_XPU BSDFHelper() = default;

            LM_XPU explicit BSDFHelper(FresnelType fresnel_type)
                    : _fresnel_type(fresnel_type) {}

            LM_XPU BSDFHelper(float4 params, float3 val, FresnelType fresnel_type)
                    : _params(params), val0(val.x), val1(val.y), val2(val.z), _fresnel_type(fresnel_type) {}

            /**
             * for metal
             * @return
             */
            ND_XPU_INLINE Spectrum metal_eta() const {
                switch (_fresnel_type) {
                    case NoOp:
                    case Dielectric:
                    case DisneyFr:
                        break;
                    case Conductor:
                        return _params;
                }
                DCHECK(0);
                return {1.f};
            }

            ND_XPU_INLINE float roughness() const {
                // todo
                return 1.f;
            }

            /**
             * for metal
             * @return
             */
            ND_XPU_INLINE float4 k() const {
                return make_float4(val0, val1, val2, 1.f);
            }

            /**
             * for dielectric material
             * @return
             */
            ND_XPU_INLINE float eta() const {
                return _params.w;
            }

            /**
             * for oren nayar bsdf
             * @return
             */
            ND_XPU_INLINE float2 AB() const {
                return make_float2(_params);
            }

            LM_XPU_INLINE void correct_eta(float cos_theta) {
                switch (_fresnel_type) {
                    case DisneyFr:
                    case FresnelType::Dielectric: {
                        _params.w = luminous::correct_eta(cos_theta, _params.w);
                        break;
                    }
                    case FresnelType::NoOp:
                    case FresnelType::Conductor:
                    default:
                        break;
                }
            }

            ND_XPU_INLINE FresnelType type() const { return _fresnel_type; }

            ND_XPU_INLINE Spectrum eval_fresnel(float cos_theta) const {
                switch (_fresnel_type) {
                    case NoOp:
                        return {1.f};
                    case Conductor:
                        return fresnel_complex(cos_theta, Spectrum(metal_eta()), Spectrum(k()));
                    case Dielectric:
                        return fresnel_dielectric(cos_theta, eta());
//                    case DisneyFr:
//                        return lerp(metallic(),
//                                    Spectrum{fresnel_dielectric(cos_theta, eta())},
//                                    fresnel_schlick(R0(), cos_theta));
                    default:
                        DCHECK(0);
                }
                return {0.f};
            }
        };

        class PhysicallyMaterialData {
        private:
            // eta for metal
            float4 _params{};

            // k for metal
            float val0;
            float val1;
            float val2;
            FresnelType _fresnel_type{NoOp};
        public:

            LM_XPU PhysicallyMaterialData() = default;

            LM_XPU explicit PhysicallyMaterialData(FresnelType fresnel_type)
                    : _fresnel_type(fresnel_type) {}

            LM_XPU PhysicallyMaterialData(float4 params, float3 val, FresnelType fresnel_type)
                    : _params(params), val0(val.x), val1(val.y), val2(val.z), _fresnel_type(fresnel_type) {}

            ND_XPU_INLINE float3 params1() const {
                return make_float3(val0, val1, val2);
            }

            ND_XPU_INLINE BSDFHelper get_helper() const {
                auto ret = BSDFHelper{_params, params1(), _fresnel_type};
                return ret;
            }

            LM_ND_XPU static PhysicallyMaterialData create_metal_data(float4 eta, float4 k) {
                PhysicallyMaterialData ret{eta, make_float3(k), Conductor};
                return ret;
            }

            LM_ND_XPU static PhysicallyMaterialData create_fake_metal_data(float4 color) {
                PhysicallyMaterialData ret{NoOp};
                return ret;
            }

            LM_ND_XPU static PhysicallyMaterialData create_mirror_data(float4 color) {
                PhysicallyMaterialData ret{NoOp};
                return ret;
            }

            LM_ND_XPU static PhysicallyMaterialData create_oren_nayar_data(float4 color, float sigma) {
                PhysicallyMaterialData ret{NoOp};
                sigma = radians(sigma);
                float sigma2 = sqr(sigma);
                float A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
                float B = 0.45f * sigma2 / (sigma2 + 0.09f);
                ret._params = make_float4(A, B, 0, 0);
                return ret;
            }

            LM_ND_XPU static PhysicallyMaterialData create_diffuse_data(float4 color) {
                PhysicallyMaterialData ret{NoOp};
                return ret;
            }

            LM_ND_XPU static PhysicallyMaterialData create_glass_data(float4 color, float eta) {
                PhysicallyMaterialData ret{Dielectric};
                ret._params.w = eta;
                return ret;
            }

        };

//        class DisneyMaterialData {
//        private:
//            // disney params
//            float4 _color{};
//            float4 _color_tint{};
//            float4 _color_sheen_tint{};
//            float _eta;
//            float _metallic{};
//            float _roughness{};
//            float _specular_tint{};
//            float _anisotropic{};
//            float _sheen{};
//            float _sheen_tint{};
//            float _clearcoat{};
//            float _clearcoat_gloss{};
//            float _spec_trans{};
//            float4 _scatter_distance{};
//            float _flatness{};
//            float _diff_trans{};
//        public:
//            ND_XPU_INLINE BSDFHelper get_helper() const {
//                return BSDFHelper();
//            }
//        };
    }
}