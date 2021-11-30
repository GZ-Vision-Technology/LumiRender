//
// Created by Zero on 30/11/2021.
//


#pragma once

#include "base_libs/optics/optics.h"

namespace luminous {
    inline namespace render {

        struct FresnelDielectric {
        private:
            float _eta_i, _eta_t;
        public:
            LM_XPU FresnelDielectric() = default;

            LM_XPU explicit FresnelDielectric(float eta_t, float eta_i = 1.f)
                    : _eta_i(eta_i), _eta_t(eta_t) {}

            LM_ND_XPU float eval(float cos_theta_i) const {
                return fresnel_dielectric(cos_theta_i, _eta_i, _eta_t);
            }
        };

        struct FresnelConductor {
        private:
            Spectrum _eta_i, _eta_t, _k;
        public:
            LM_XPU FresnelConductor() = default;

            LM_XPU FresnelConductor(Spectrum eta_t, Spectrum k, Spectrum eta_i = Spectrum{1.f})
                    : _eta_i(eta_i), _eta_t(eta_t), _k(k) {}

            LM_ND_XPU Spectrum eval(float cos_theta_i) const {
                return fresnel_conductor(cos_theta_i, _eta_i, _eta_t, _k);
            }
        };
    }
}