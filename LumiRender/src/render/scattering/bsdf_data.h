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

        struct BSDFHelper {
        private:
            /**
             * @brief variant parameter, must be one of the following layouts
             *  - A B for oren nayar
             *  - R0 for disney _params.w is clearcoat alpha for disney
             */
            float4 _params{};

            /**
             * @brief variant parameter, must be one of the following layouts
             *  - roughness for disney
             *  - k0 for metal
             */
            float val0{};

            /**
             * @brief variant parameter, must be one of the following layouts
             *  - metallic for disney
             *  - k1 for metal
             */
            float val1{};

            /**
             * @brief variant parameter, must be one of the following layouts
             *  - eta for dielectric
             *  - k2 for metal
             */
            float val2{};

            /**
             * @brief Fresnel formula type.
             */
            FresnelType _fresnel_type{NoOp};
        public:
            LM_XPU BSDFHelper() = default;

            LM_XPU explicit BSDFHelper(FresnelType fresnel_type)
                    : _fresnel_type(fresnel_type) {}

            LM_XPU BSDFHelper(float4 params, float3 val, FresnelType fresnel_type)
                    : _params(params), val0(val.x), val1(val.y), val2(val.z), _fresnel_type(fresnel_type) {}

            ND_XPU_INLINE BSDFHelper get_helper() const {
                return *this;
            }

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

            /**
             * for disney material
             * @return
             */
            ND_XPU_INLINE Spectrum R0() const {
                return _params;
            }

            LM_XPU_INLINE void set_R0(Spectrum val) {
                _params = make_float4(val.vec(), _params.w);
            }

            /**
             * for disney material
             * @return
             */
            ND_XPU_INLINE float clearcoat_alpha() const {
                return _params.w;
            }

            LM_XPU_INLINE void set_clearcoat_alpha(float alpha) {
                _params.w = alpha;
            }

            /**
             * for disney material
             * @return
             */
            ND_XPU_INLINE float metallic() const {
                return val1;
            }

            LM_XPU_INLINE void set_metallic(float m) {
                val1 = m;
            }

            /**
             * for disney material
             * @return
             */
            ND_XPU_INLINE float roughness() const {
                return val0;
            }

            LM_XPU_INLINE void set_roughness(float rough) {
                val0 = rough;
            }


            /**
             * for dielectric material
             * @return
             */
            ND_XPU_INLINE float eta() const {
                return val2;
            }

            LM_XPU_INLINE void set_eta(float e) {
                val2 = e;
            }

            /**
             * for metal
             * @return
             */
            ND_XPU_INLINE float4 k() const {
                return make_float4(val0, val1, val2, 1.f);
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
                        val2 = luminous::correct_eta(cos_theta, val2);
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
                    case DisneyFr:
                        return lerp(metallic(),
                                    Spectrum{fresnel_dielectric(cos_theta, eta())},
                                    fresnel_schlick(R0(), cos_theta));
                    default:
                        DCHECK(0);
                }
                return {0.f};
            }

            LM_ND_XPU static BSDFHelper create_substrate_data(float eta) {
                BSDFHelper ret{Dielectric};
                ret.val2 = eta;
                return ret;
            }

            LM_ND_XPU static BSDFHelper create_metal_data(float4 eta, float3 k) {
                BSDFHelper ret{eta, make_float3(k), Conductor};
                return ret;
            }

            LM_ND_XPU static BSDFHelper create_fake_metal_data(float3 color) {
                BSDFHelper ret{NoOp};
                return ret;
            }

            LM_ND_XPU static BSDFHelper create_mirror_data(float3 color) {
                BSDFHelper ret{NoOp};
                return ret;
            }

            LM_ND_XPU static BSDFHelper create_oren_nayar_data(float3 color, float sigma) {
                BSDFHelper ret{NoOp};
                sigma = radians(sigma);
                float sigma2 = sqr(sigma);
                float A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
                float B = 0.45f * sigma2 / (sigma2 + 0.09f);
                ret._params = make_float4(A, B, 0, 0);
                return ret;
            }

            LM_ND_XPU static BSDFHelper create_diffuse_data(float3 color) {
                BSDFHelper ret{NoOp};
                return ret;
            }

            LM_ND_XPU static BSDFHelper create_glass_data(float3 color, float eta) {
                BSDFHelper ret{Dielectric};
                ret.val2 = eta;
                return ret;
            }
        };
    }
}