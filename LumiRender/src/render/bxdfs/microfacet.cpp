//
// Created by Zero on 29/11/2021.
//

#include "microfacet.h"

namespace luminous {
    inline namespace render {

        float MicrofacetDistribution::D(const float3 &wh) const {
            return 0;
        }

        float MicrofacetDistribution::lambda(const float3 &w) const {
            return 0;
        }

        float3 MicrofacetDistribution::sample_wh(const float3 &wo, const float2 &u) const {
            return luminous::float3();
        }

        float MicrofacetDistribution::PDF_dir(const float3 &wo, const float3 &wh) const {
            return 0;
        }
    }
}