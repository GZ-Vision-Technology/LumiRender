//
// Created by Zero on 30/11/2021.
//


#pragma once

#include "base_libs/optics/optics.h"
#include "bsdf_data.h"

namespace luminous {
    inline namespace render {

        struct FresnelDielectric {
        public:
            float eta{};
            LM_XPU FresnelDielectric() = default;

            LM_XPU explicit FresnelDielectric(float eta_t)
                    : eta(eta_t) {}

            LM_XPU void correct_eta(float cos_theta) {
                eta = luminous::correct_eta(cos_theta, eta);
            }

            LM_ND_XPU float eval(float abs_cos_theta_i) const {
                return fresnel_dielectric(abs_cos_theta_i, eta);
            }
        };

        struct FresnelConductor {
        public:
            Spectrum eta, k;
            LM_XPU FresnelConductor() = default;

            LM_XPU FresnelConductor(Spectrum eta, Spectrum k)
                    : eta(eta), k(k) {}

            LM_XPU void correct_eta(float cos_theta) {
                eta = luminous::correct_eta(cos_theta, eta);
            }

            LM_ND_XPU Spectrum eval(float cos_theta_i) const {
                return fresnel_complex(cos_theta_i, eta, k);
            }
        };

        struct FresnelNoOp {
        public:
            constexpr static float eta{1.f};

            LM_XPU FresnelNoOp() = default;

            LM_XPU void correct_eta(float cos_theta) {}

            LM_ND_XPU float eval(float) const {
                return 1.f;
            }
        };


        struct Fresnel {
        private:
            FresnelType _type{NoOp};
        public:

            LM_XPU explicit Fresnel(FresnelType type = NoOp) : _type(type) {}

            ND_XPU_INLINE FresnelType type() const { return _type; }

            ND_XPU_INLINE Spectrum eval(float cos_theta, BSDFData data) const {
                switch (_type) {
                    case NoOp:
                        return {1.f};
                    case Metal:
                        return fresnel_complex(cos_theta, Spectrum(data.metal_eta()), Spectrum(data.k()))[0];
                    case Dielectric:
                        return fresnel_dielectric(cos_theta, data.eta());
                    default:
                        DCHECK(0);
                }
                return {0.f};
            }
        };
    }
}