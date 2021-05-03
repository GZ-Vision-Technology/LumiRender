//
// Created by Zero on 2021/5/1.
//


#pragma once

#include "render/include/interaction.h"
#include "render/bxdfs/bxdf.h"

namespace luminous {
    inline namespace render {

        class BSDF {
        private:
            BxDF _bxdf;
            float3 _ng;
            Frame _shading_frame;
        public:
            XPU BSDF() = default;

            XPU BSDF(float3 ng, float3 ns, float3 dp_dus, BxDF bxdf)
                    : _ng(ng), _shading_frame(Frame::from_xz(dp_dus, ns)), _bxdf(bxdf) {}

            NDSC_XPU float4 eval(float3 wo_world, float3 wi_world,
                                 TransportMode mode = TransportMode::Radiance) const {
                float3 wo = _shading_frame.to_local(wo_world);
                float3 wi = _shading_frame.to_local(wi_world);
                if (wo.z == 0) {
                    return make_float4(0);
                }
                return _bxdf.eval(wo, wi, mode);
            }

            NDSC_XPU float4 base_color() const {
                return _bxdf.base_color();
            }

            NDSC_XPU float4 rho_hd(float3 wo_world, BufferView<const float> uc,
                                   BufferView<const float2> u2) const {
                float3 wo = to_local(wo_world);
                return _bxdf.rho_hd(wo, uc, u2);
            }

            NDSC_XPU float4 rho_hh(BufferView<const float2> u1, BufferView<const float> uc,
                                   BufferView<const float2> u2) const {
                return _bxdf.rho_hh(u1, uc, u2);
            }

            NDSC_XPU float3 to_local(float3 val) const {
                return _shading_frame.to_local(val);
            }

            NDSC_XPU float3 to_world(float3 val) const {
                return _shading_frame.to_world(val);
            }

            NDSC_XPU bool is_non_specular() const {
                return luminous::is_non_specular(_bxdf.flags());
            }

            NDSC_XPU bool is_reflective() const {
                return luminous::is_reflective(_bxdf.flags());
            }

            NDSC_XPU bool is_transmissive() const {
                return luminous::is_transmissive(_bxdf.flags());
            }

            NDSC_XPU bool is_diffuse() const {
                return luminous::is_diffuse(_bxdf.flags());
            }

            NDSC_XPU bool is_glossy() const {
                return luminous::is_glossy(_bxdf.flags());
            }

            NDSC_XPU bool is_specular() const {
                return luminous::is_specular(_bxdf.flags());
            }
        };

    } // luminous::render
} // luminous