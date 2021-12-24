//
// Created by Zero on 16/12/2021.
//


#pragma once

#include "base_libs/optics/rgb.h"

namespace luminous {
    inline namespace render {

        enum FresnelType : uint8_t {
            NoOp,
            Metal,
            Dielectric
        };

        struct BSDFData {
        private:
            float4 _color{};
            float4 _params{};
        public:

            LM_XPU BSDFData() = default;

            /**
             * for metal
             * @return
             */
            ND_XPU_INLINE float4 metal_eta() const {
                return _color;
            }

            ND_XPU_INLINE float4 color() const {
                return _color;
            }

            /**
             * for metal
             * @return
             */
            ND_XPU_INLINE float4 k() const {
                return _params;
            }

            /**
             * for dielectric material
             * @return
             */
            ND_XPU_INLINE float eta() const {
                return _params.w;
            }

            /**
             * for plastic material
             * @return
             */
            ND_XPU_INLINE float4 spec() const {
                return _params;
            }

            /**
             * for oren nayar bsdf
             * @return
             */
            ND_XPU_INLINE float2 AB() const {
                return make_float2(_params);
            }

            LM_XPU_INLINE void correct_eta(float cos_theta, FresnelType fresnel_type) {
                switch (fresnel_type) {
                    case FresnelType::Dielectric: {
                        _params.w = rcp(_params.w);
                        break;
                    }
                    default:
                        DCHECK(0);
                }
            }

            LM_ND_XPU static BSDFData create_metal_data(float4 eta, float4 k) {
                BSDFData ret{};
                ret._color = eta;
                ret._params = k;
                return ret;
            }

            LM_ND_XPU static BSDFData create_mirror_data(float4 color) {
                BSDFData ret{};
                ret._color = color;
                return ret;
            }

            LM_ND_XPU static BSDFData create_oren_nayar_data(float4 color, float sigma) {
                BSDFData ret;
                ret._color = color;
                sigma = radians(sigma);
                float sigma2 = sqr(sigma);
                float A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
                float B = 0.45f * sigma2 / (sigma2 + 0.09f);
                ret._params = make_float4(A, B, 0, 0);
                return ret;
            }

            LM_ND_XPU static BSDFData create_diffuse_data(float4 color) {
                BSDFData ret;
                ret._color = color;
                return ret;
            }

            LM_ND_XPU static BSDFData create_glass_data(float4 color, float eta) {
                BSDFData ret;
                ret._color = color;
                ret._params.w = eta;
                return ret;
            }

            LM_ND_XPU static BSDFData create_plastic_data(float4 color, float4 spec, float eta) {
                BSDFData ret;
                ret._color = color;
                ret._params = spec;
                ret._params.w = eta;
                return ret;
            }
        };
    }
}