//
// Created by Zero on 30/11/2021.
//


#pragma once

#include "base_libs/optics/optics.h"
#include "bsdf_data.h"

namespace luminous {
    inline namespace render {

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
                    case Conductor:
                        return fresnel_complex(cos_theta, Spectrum(data.metal_eta()), Spectrum(data.k()));
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