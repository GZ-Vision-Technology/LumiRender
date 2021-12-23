//
// Created by Zero on 16/12/2021.
//


#pragma once

#include "base_libs/optics/rgb.h"

namespace luminous {
    inline namespace render {

        struct MetalData {
            float4 metal_eta{};
            float4 k{};

            LM_XPU MetalData(float4 eta, float4 k)
                    : metal_eta(eta), k(k) {}
        };

        struct BSDFBaseData {
            float4 color{0.f};

            LM_XPU explicit BSDFBaseData(float4 color)
                    : color(color) {}
        };

        struct PlasticData : BSDFBaseData {
            float4 spec{make_float4(1.f)};
            LM_XPU PlasticData(float4 color, float4 spec)
                    : BSDFBaseData(color), spec(spec) {}
        };

        struct GlassData : public BSDFBaseData {
            float eta{};
            LM_XPU GlassData(float4 color, float eta)
                    : BSDFBaseData(color), eta(eta) {}
        };

        struct DiffuseData : public BSDFBaseData {
            float A{};
            float B{};

            LM_XPU DiffuseData(float4 color, float sigma)
                    : BSDFBaseData(color) {
                sigma = radians(sigma);
                float sigma2 = sqr(sigma);
                A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
                B = 0.45f * sigma2 / (sigma2 + 0.09f);
            }
        };

        using MirrorData = BSDFBaseData;

        struct BSDFData {
            LM_XPU BSDFData() {}

            union {
                MetalData metal_data;
                DiffuseData diffuse_data;
                GlassData glass_data;
                PlasticData plastic_data;
                MirrorData mirror_data;
            };

            LM_ND_XPU static BSDFData create_metal_data(float4 eta, float4 k) {
                BSDFData ret{};
                ret.metal_data = {eta, k};
                return ret;
            }

            LM_ND_XPU static BSDFData create_mirror_data(float4 color) {
                BSDFData ret{};
                ret.mirror_data.color = color;
                return ret;
            }

            LM_ND_XPU static BSDFData create_oren_nayar_data(float4 color, float sigma) {
                BSDFData ret;
                ret.diffuse_data = {color, sigma};
                return ret;
            }

            LM_ND_XPU static BSDFData create_diffuse_data(float4 color) {
                BSDFData ret;
                ret.diffuse_data.color = color;
                return ret;
            }

            LM_ND_XPU static BSDFData create_glass_data(float4 color, float eta) {
                BSDFData ret;
                ret.glass_data = {color, eta};
                return ret;
            }

            LM_ND_XPU static BSDFData create_plastic_data(float4 color, float4 spec) {
                BSDFData ret;
                ret.plastic_data = {color, spec};
                return ret;
            }
        };

        struct BSDFCommonData {
            float4 color{0.f};

            LM_XPU explicit BSDFCommonData(float4 color)
                    : color(color) {}
        };


        struct OrenNayarData : public BSDFCommonData {
            float A{};
            float B{};

            LM_XPU OrenNayarData(float4 color, float sigma)
                    : BSDFCommonData(color) {
                sigma = radians(sigma);
                float sigma2 = sqr(sigma);
                A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
                B = 0.45f * sigma2 / (sigma2 + 0.09f);
            }
        };
    }
}