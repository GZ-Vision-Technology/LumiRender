//
// Created by Zero on 2021/5/13.
//

#include "bsdf_wrapper.h"

namespace luminous {
    inline namespace render {
        Spectrum BSDFWrapper::eval(float3 wo_world, float3 wi_world,
                                   BxDFFlags sample_flags,
                                   TransportMode mode) const {
            // todo test single sample_flags
            float3 wo = to_local(wo_world);
            float3 wi = to_local(wi_world);
            // todo normal map may lead light leak
            return _bsdf.eval(wo, wi, sample_flags, mode) * abs_dot(_shading_frame.z, wi_world);
        }

        float BSDFWrapper::PDF(float3 wo_world, float3 wi_world,
                               BxDFFlags sample_flags,
                               TransportMode mode) const {
            float3 wo = to_local(wo_world);
            float3 wi = to_local(wi_world);
            // todo normal map may lead light leak
            return _bsdf.PDF(wo, wi, sample_flags, mode);
        }

        BSDFSample BSDFWrapper::sample_f(float3 world_wo, float uc, float2 u,
                                         BxDFFlags sample_flags,
                                         TransportMode mode) const {
            float3 local_wo = to_local(world_wo);
            BSDFSample ret = _bsdf.sample_f(local_wo, uc, u, sample_flags, mode);
            ret.wi = to_world(ret.wi);
            bool sh = same_hemisphere(world_wo, ret.wi, _ng);
            // todo normal map may lead light leak ,change to bump map
            if (ret.valid()) {
                ret.f_val *= abs_dot(_shading_frame.z, ret.wi);
            }
            return ret;
        }

        Spectrum BSDFWrapper::color() const {
            return _bsdf.color();
        }
    } // luminous::render
} // luminous