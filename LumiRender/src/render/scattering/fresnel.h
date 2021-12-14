//
// Created by Zero on 30/11/2021.
//


#pragma once

#include "base_libs/optics/optics.h"

namespace luminous {
    inline namespace render {

        struct FresnelDielectric {
        private:
            float _eta{};
        public:
            LM_XPU FresnelDielectric() = default;

            LM_XPU explicit FresnelDielectric(float eta_t)
                    : _eta(eta_t) {}

            LM_ND_XPU float eval(float cos_theta_i) const {
                return fresnel_dielectric(cos_theta_i, _eta);
            }
        };

        struct FresnelConductor {
        private:
            Spectrum _eta, _k;
        public:
            LM_XPU FresnelConductor() = default;

            LM_XPU FresnelConductor(Spectrum eta, Spectrum k)
                    : _eta(eta), _k(k) {}

            LM_ND_XPU Spectrum eval(float cos_theta_i) const {
                return fresnel_complex(cos_theta_i, _eta, _k);
            }
        };

        struct FresnelNoOp {
        public:
            LM_XPU FresnelNoOp() = default;

            LM_ND_XPU float eval(float ) const {
                return 1.f;
            }
        };
    }
}