//
// Created by Zero on 2021/4/7.
//

#include "light_handle.h"

namespace luminous {
    inline namespace render {

        LightType LightHandle::type() const {
            LUMINOUS_VAR_DISPATCH(type);
        }

        bool LightHandle::is_delta() const {
            LUMINOUS_VAR_DISPATCH(is_delta);
        }

        float3 LightHandle::sample_Li(DirectSamplingRecord *rcd, float2 u) const {
            LUMINOUS_VAR_DISPATCH(sample_Li, rcd, u);
        }

        float LightHandle::PDF_Li(const DirectSamplingRecord &rcd) const {
            LUMINOUS_VAR_DISPATCH(PDF_Li, rcd);
        }

        float3 LightHandle::power() const {
            LUMINOUS_VAR_DISPATCH(power);
        }

        std::string LightHandle::to_string() const {
            LUMINOUS_VAR_DISPATCH(to_string);
        }
    } // luminous::render
} // luminous