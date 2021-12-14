//
// Created by Zero on 14/12/2021.
//


#pragma once

#include "base.h"
#include "fresnel.h"

namespace luminous {
    inline namespace render {

        template<typename Fresnel>
        class SpecularReflection {
        private:
            float3 _r;
            Fresnel _fresnel;
        public:
            LM_XPU SpecularReflection(float3 r, Fresnel fresnel)
                : _r(r), _fresnel(fresnel) {}


        };
    }
}