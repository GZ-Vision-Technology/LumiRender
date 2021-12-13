//
// Created by Zero on 2021/5/1.
//


#pragma once

#include "render/bxdfs/bxdf.h"
#include "base_libs/optics/rgb.h"
#include "render/integrators/wavefront/soa.h"

namespace luminous {
    inline namespace render {

        class BSDFWrapper {
        private:
            BxDF _bxdf;
            float3 _ng;
            Frame _shading_frame;
        public:
            LM_XPU BSDFWrapper() = default;

            LM_XPU BSDFWrapper(float3 ng, float3 ns, float3 dp_dus, BxDF bxdf)
                    : _ng(ng), _shading_frame(Frame::from_xz(dp_dus, ns)), _bxdf(std::move(bxdf)) {}

            LM_ND_XPU Spectrum eval(float3 wo_world, float3 wi_world,
                                    TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU float PDF(float3 wo_world, float3 wi_world,
                                TransportMode mode = TransportMode::Radiance,
                                BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const;

            LM_ND_XPU lstd::optional<BSDFSample> sample_f(float3 world_wo, float uc, float2 u,
                                                          TransportMode mode = TransportMode::Radiance,
                                                          BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const;

            LM_ND_XPU float4 base_color() const;

            LM_ND_XPU Spectrum rho_hd(float3 wo_world, BufferView<const float> uc,
                                      BufferView<const float2> u2) const;

            LM_ND_XPU Spectrum rho_hh(BufferView<const float2> u1, BufferView<const float> uc,
                                      BufferView<const float2> u2) const;

            LM_ND_XPU float3 to_local(float3 val) const {
                return _shading_frame.to_local(val);
            }

            LM_ND_XPU float3 to_world(float3 val) const {
                return _shading_frame.to_world(val);
            }

            LM_ND_XPU bool is_non_specular() const {
                return luminous::is_non_specular(_bxdf.flags());
            }

            LM_ND_XPU bool is_reflective() const {
                return luminous::is_reflective(_bxdf.flags());
            }

            LM_ND_XPU bool is_transmissive() const {
                return luminous::is_transmissive(_bxdf.flags());
            }

            LM_ND_XPU bool is_diffuse() const {
                return luminous::is_diffuse(_bxdf.flags());
            }

            LM_ND_XPU bool is_glossy() const {
                return luminous::is_glossy(_bxdf.flags());
            }

            LM_ND_XPU bool is_specular() const {
                return luminous::is_specular(_bxdf.flags());
            }
        };

    } // luminous::render
} // luminous