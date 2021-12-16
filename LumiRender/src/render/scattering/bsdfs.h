//
// Created by Zero on 16/12/2021.
//


#pragma once

#include "bsdf_ty.h"
#include "microfacet.h"
#include "bsdf_data.h"
#include "fresnel.h"
#include "diffuse_scatter.h"

namespace luminous {
    inline namespace render {

        using DiffuseBSDF = BSDF_Ty<DiffuseData, MicrofacetNone, FresnelNoOp, DiffuseReflection>;

        class BSDF : public Variant<DiffuseBSDF> {
        private:
            using Variant::Variant;
        public:
            GEN_BASE_NAME(BSDF)


        };
    }
}