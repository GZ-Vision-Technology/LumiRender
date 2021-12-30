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

        ND_XPU_INLINE MicrofacetType get_microfacet_type(FresnelType fresnel_type) {
            MicrofacetType array[4] = {None, GGX, Disney, GGX};
            return array[uint8_t(fresnel_type)];
        }

        class DisneyMaterialData;

        class PhysicallyMaterialData;

        struct BSDFParam {
        private:
            // todo Merge field
            float4 _color{};
            float4 _params{};
            FresnelType _fresnel_type{NoOp};
        public:

            MicrofacetDistrib microfacet{};

            LM_XPU BSDFParam() = default;

            LM_XPU explicit BSDFParam(FresnelType fresnel_type)
                    : _fresnel_type(fresnel_type) {}

            LM_XPU BSDFParam(const float4 color, const float4 params, const FresnelType type)
                    : _color(color), _params(params), _fresnel_type(type) {}

            ND_XPU_INLINE MicrofacetType microfacet_type() const {
                return get_microfacet_type(_fresnel_type);
            }

            /**
             * for metal
             * @return
             */
            ND_XPU_INLINE float4 metal_eta() const {
                switch (_fresnel_type) {
                    case Conductor:
                        return _color;
                }
                DCHECK(0);
                return make_float4(1.f);
            }

            ND_XPU_INLINE BSDFParam get_param() const {
                return *this;
            }

            ND_XPU_INLINE float4 color() const {
                switch (_fresnel_type) {
                    case NoOp:
                    case Dielectric:
                        return _color;
                    case Conductor:
                        return make_float4(1.f);
                }
                DCHECK(0);
                return make_float4(1.f);
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

            LM_XPU_INLINE void correct_eta(float cos_theta) {
                switch (_fresnel_type) {
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
                    default:
                        DCHECK(0);
                }
                return {0.f};
            }
        };

        class PhysicallyMaterialData {
        private:
            float4 _color{};
            float4 _params{};
            FresnelType _fresnel_type{NoOp};
            float _alpha_x{};
            float _alpha_y{};
        public:
            MicrofacetDistrib microfacet{};

            LM_XPU PhysicallyMaterialData() = default;

            LM_XPU explicit PhysicallyMaterialData(FresnelType fresnel_type)
                    : _fresnel_type(fresnel_type) {}

            ND_XPU_INLINE BSDFParam get_param() const {
                auto ret = BSDFParam{_color, _params, _fresnel_type};
                ret.microfacet = MicrofacetDistrib{_alpha_x, _alpha_y, get_microfacet_type(_fresnel_type)};
                return ret;
            }

            LM_ND_XPU static PhysicallyMaterialData create_metal_data(float4 eta, float4 k,float alpha_x, float alpha_y) {
                PhysicallyMaterialData ret{Conductor};
                ret._color = eta;
                ret._params = k;
                ret._alpha_x = alpha_x;
                ret._alpha_y = alpha_y;
                return ret;
            }

            LM_ND_XPU static PhysicallyMaterialData create_fake_metal_data(float4 color,float alpha_x, float alpha_y) {
                PhysicallyMaterialData ret{NoOp};
                ret._color = color;
                ret._alpha_x = alpha_x;
                ret._alpha_y = alpha_y;
                return ret;
            }

            LM_ND_XPU static PhysicallyMaterialData create_mirror_data(float4 color) {
                PhysicallyMaterialData ret{NoOp};
                ret._color = color;
                return ret;
            }

            LM_ND_XPU static PhysicallyMaterialData create_oren_nayar_data(float4 color, float sigma) {
                PhysicallyMaterialData ret{NoOp};
                ret._color = color;
                sigma = radians(sigma);
                float sigma2 = sqr(sigma);
                float A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
                float B = 0.45f * sigma2 / (sigma2 + 0.09f);
                ret._params = make_float4(A, B, 0, 0);
                return ret;
            }

            LM_ND_XPU static PhysicallyMaterialData create_diffuse_data(float4 color) {
                PhysicallyMaterialData ret{NoOp};
                ret._color = color;
                return ret;
            }

            LM_ND_XPU static PhysicallyMaterialData create_glass_data(float4 color, float eta, float alpha_x, float alpha_y) {
                PhysicallyMaterialData ret{Dielectric};
                ret._color = color;
                ret._params.w = eta;
                ret._alpha_x = alpha_x;
                ret._alpha_y = alpha_y;
                return ret;
            }

            LM_ND_XPU static PhysicallyMaterialData create_plastic_data(float4 color, float4 spec, float eta = 1.5f) {
                PhysicallyMaterialData ret{Dielectric};
                ret._color = color;
                ret._params = spec;
                ret._params.w = eta;
                return ret;
            }
        };

        class DisneyMaterialData {
        private:
            // disney params
            float4 color{};
            float eta;
            float metallic{};
            float roughness{};
            float specular_tint{};
            float anisotropic{};
            float sheen{};
            float sheen_tint{};
            float clearcoat{};
            float clearcoat_gloss{};
            float spec_trans{};
            float4 scatter_distance{};
            float flatness{};
            float diff_trans{};
            bool thin{};
        };
    }
}