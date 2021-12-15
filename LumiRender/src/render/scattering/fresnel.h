//
// Created by Zero on 30/11/2021.
//


#pragma once

#include "base_libs/optics/optics.h"

namespace luminous {
    inline namespace render {

        struct FresnelDielectric {
        public:
            float eta{};
            LM_XPU FresnelDielectric() = default;

            LM_XPU explicit FresnelDielectric(float eta_t)
                    : eta(eta_t) {}

            LM_ND_XPU float eval(float cos_theta_i) const {
                return fresnel_dielectric(cos_theta_i, eta);
            }
        };

        struct FresnelConductor {
        public:
            Spectrum eta, k;
            LM_XPU FresnelConductor() = default;

            LM_XPU FresnelConductor(Spectrum eta, Spectrum k)
                    : eta(eta), k(k) {}

            LM_ND_XPU Spectrum eval(float cos_theta_i) const {
                return fresnel_complex(cos_theta_i, eta, k);
            }
        };

        struct FresnelNoOp {
        public:
            constexpr static float eta = 1.f;

            LM_XPU FresnelNoOp() = default;

            LM_ND_XPU float eval(float) const {
                return 1.f;
            }
        };
    }
}