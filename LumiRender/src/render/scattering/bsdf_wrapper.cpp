//
// Created by Zero on 2021/5/13.
//

#include "bsdf_wrapper.h"

namespace luminous {
    inline namespace render {
        Spectrum BSDFWrapper::eval(float3 wo_world, float3 wi_world, TransportMode mode) const {
            float3 wo = to_local(wo_world);
            float3 wi = to_local(wi_world);

            if (wo.z == 0) {
                return Spectrum{0};
            }
            return _bsdf.eval(wo, wi) * abs_dot(_shading_frame.z, wi_world);
        }

        float BSDFWrapper::PDF(float3 wo_world, float3 wi_world, TransportMode mode, BxDFReflTransFlags sample_flags) const {
            float3 wo = to_local(wo_world);
            float3 wi = to_local(wi_world);
            return _bsdf.PDF(wo, wi);
        }

        BSDFSample BSDFWrapper::sample_f(float3 world_wo, float uc, float2 u,
                                                         TransportMode mode, BxDFReflTransFlags sample_flags) const {
            float3 local_wo = to_local(world_wo);
            BSDFSample ret = _bsdf.sample_f(local_wo, uc, u);
            if (ret.valid()) {
                ret.wi = to_world(ret.wi);
                ret.f_val *= abs_dot(_shading_frame.z, ret.wi);
            }
            return ret;
        }

        Spectrum BSDFWrapper::base_color() const {
            return _bsdf.color();
        }
    } // luminous::render
} // luminous