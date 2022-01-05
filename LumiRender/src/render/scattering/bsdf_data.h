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
            MicrofacetType array[4] = {GGX, GGX, Disney, GGX};
            return array[uint8_t(fresnel_type)];
        }

        class DisneyMaterialData;

        class PhysicallyMaterialData;

        struct BSDFHelper {
        private:
            // todo Merge field
            float4 _color{};

            // color_tint, eta for disney
            float4 _params{};

            // color_sheen_tint disney
            float4 _params1{};

            // clearcoat gloss diff_trans spec_trans for disney
            float4 _params2{};

            // R0, metallic for disney
            float4 _params3{};

            FresnelType _fresnel_type{NoOp};

            friend class DisneyMaterialData;
        public:
            LM_XPU BSDFHelper() = default;

            LM_XPU explicit BSDFHelper(FresnelType fresnel_type)
                    : _fresnel_type(fresnel_type) {}

            LM_XPU BSDFHelper(const float4 color, const float4 params, const FresnelType type)
                    : _color(color), _params(params), _fresnel_type(type) {}

            LM_XPU BSDFHelper(const float4 color, const float4 params,
                              const FresnelType type, float alpha_x, float alpha_y)
                    : _color(color), _params(params), _fresnel_type(type) {}

            ND_XPU_INLINE MicrofacetType microfacet_type() const {
                return get_microfacet_type(_fresnel_type);
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
                        return _color;
                }
                DCHECK(0);
                return {1.f};
            }

            ND_XPU_INLINE Spectrum color_tint() const {
                return _params;
            }

            ND_XPU_INLINE Spectrum color_sheen_tint() const {
                return _params1;
            }

            ND_XPU_INLINE float clear_coat() const {
                return _params2.x;
            }

            ND_XPU_INLINE float gloss() const {
                return _params2.y;
            }

            ND_XPU_INLINE float diff_trans() const {
                return _params2.z;
            }

            ND_XPU_INLINE float spec_trans() const {
                return _params2.w;
            }

            ND_XPU_INLINE Spectrum R0() const {
                return Spectrum{_params3};
            }

            ND_XPU_INLINE float roughness() const {
                // todo
                return 1.f;
            }

            ND_XPU_INLINE float metallic() const {
                return _params3.w;
            }

            ND_XPU_INLINE float4 color() const {
                switch (_fresnel_type) {
                    case NoOp:
                    case DisneyFr:
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
                    case DisneyFr:
                        return lerp(metallic(),
                                    Spectrum{fresnel_dielectric(cos_theta, eta())},
                                    fresnel_schlick(R0(), cos_theta));
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

            LM_XPU PhysicallyMaterialData() = default;

            LM_XPU explicit PhysicallyMaterialData(FresnelType fresnel_type)
                    : _fresnel_type(fresnel_type) {}

            ND_XPU_INLINE BSDFHelper get_helper() const {
                auto ret = BSDFHelper{_color, _params, _fresnel_type, _alpha_x, _alpha_y};
                return ret;
            }

            LM_ND_XPU static PhysicallyMaterialData create_metal_data(float4 eta, float4 k,
                                                                      float alpha_x, float alpha_y) {
                PhysicallyMaterialData ret{Conductor};
                ret._color = eta;
                ret._params = k;
                ret._alpha_x = alpha_x;
                ret._alpha_y = alpha_y;
                return ret;
            }

            LM_ND_XPU static PhysicallyMaterialData create_fake_metal_data(float4 color, float alpha_x, float alpha_y) {
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

            LM_ND_XPU static PhysicallyMaterialData
            create_glass_data(float4 color, float eta, float alpha_x, float alpha_y) {
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
            float4 _color{};
            float4 _color_tint{};
            float4 _color_sheen_tint{};
            float _eta;
            float _metallic{};
            float _roughness{};
            float _specular_tint{};
            float _anisotropic{};
            float _sheen{};
            float _sheen_tint{};
            float _clearcoat{};
            float _clearcoat_gloss{};
            float _spec_trans{};
            float4 _scatter_distance{};
            float _flatness{};
            float _diff_trans{};
        public:
            ND_XPU_INLINE BSDFHelper get_helper() const {
                return BSDFHelper();
            }
        };
    }
}