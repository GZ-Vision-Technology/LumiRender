//
// Created by Zero on 2021/5/13.
//

#include "bsdf.h"

namespace luminous {
    inline namespace render {
        Spectrum BSDF::eval(float3 wo_world, float3 wi_world, TransportMode mode) const {
            float3 wo = to_local(wo_world);
            float3 wi = to_local(wi_world);
            if (wo.z == 0) {
                return make_float4(0);
            }
            return _bxdf.eval(wo, wi, mode) * abs_dot(_shading_frame.z, wi_world);
        }

        float BSDF::PDF(float3 wo_world, float3 wi_world, TransportMode mode, BxDFReflTransFlags sample_flags) const {
            float3 wo = to_local(wo_world);
            float3 wi = to_local(wi_world);
            return _bxdf.PDF(wo, wi, mode, sample_flags);
        }

        lstd::optional<BSDFSample> BSDF::sample_f(float3 world_wo, float uc, float2 u,
                                                  TransportMode mode, BxDFReflTransFlags sample_flags) const {
            float3 local_wo = to_local(world_wo);
            auto ret = _bxdf.sample_f(local_wo, uc, u, mode, sample_flags);
            if (ret) {
                ret->wi = to_world(ret->wi);
                ret->f_val *= abs_dot(_shading_frame.z, ret->wi);
            }
            return ret;
        }

        float4 BSDF::base_color() const {
            return _bxdf.base_color();
        }

        Spectrum BSDF::rho_hd(float3 wo_world, BufferView<const float> uc, BufferView<const float2> u2) const {
            float3 wo = to_local(wo_world);
            return _bxdf.rho_hd(wo, uc, u2);
        }

        Spectrum
        BSDF::rho_hh(BufferView<const float2> u1, BufferView<const float> uc, BufferView<const float2> u2) const {
            return _bxdf.rho_hh(u1, uc, u2);
        }
    } // luminous::render
} // luminous