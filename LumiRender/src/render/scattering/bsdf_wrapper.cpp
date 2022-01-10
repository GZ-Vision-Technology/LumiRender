//
// Created by Zero on 2021/5/13.
//

#include "bsdf_wrapper.h"

namespace luminous {
    inline namespace render {
        Spectrum BSDFWrapper::eval(float3 wo_world, float3 wi_world,
                                   BxDFFlags sample_flags,
                                   TransportMode mode) const {
            float3 wo = to_local(wo_world);
            float3 wi = to_local(wi_world);
            if (wo.z == 0 || dot(wo_world, _ng) * dot(wi_world, _ng) <= 0) {
                return Spectrum{0};
            }
            return _bsdf.eval(wo, wi) * abs_dot(_shading_frame.z, wi_world);
        }

        float BSDFWrapper::PDF(float3 wo_world, float3 wi_world,
                               BxDFFlags sample_flags,
                               TransportMode mode) const {
            float3 wo = to_local(wo_world);
            float3 wi = to_local(wi_world);
            if (dot(wo_world, _ng) * dot(wi_world, _ng) <= 0) {
                return 0.f;
            }
            return _bsdf.PDF(wo, wi, sample_flags, mode);
        }

        BSDFSample BSDFWrapper::sample_f(float3 world_wo, float uc, float2 u,
                                         BxDFFlags sample_flags,
                                         TransportMode mode) const {
            float3 local_wo = to_local(world_wo);
            BSDFSample ret = _bsdf.sample_f(local_wo, uc, u, sample_flags, mode);
            if (ret.valid()) {
                ret.wi = to_world(ret.wi);
                ret.f_val *= abs_dot(_shading_frame.z, ret.wi);
            }
            return ret;
        }

        Spectrum BSDFWrapper::color() const {
            return _bsdf.color();
        }
    } // luminous::render
} // luminous