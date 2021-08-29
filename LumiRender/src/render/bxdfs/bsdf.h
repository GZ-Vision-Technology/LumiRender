//
// Created by Zero on 2021/5/1.
//


#pragma once

#include <utility>

#include "render/bxdfs/bxdf.h"
#include "base_libs/optics/rgb.h"
#include "render/integrators/wavefront/soa.h"

namespace luminous {
    inline namespace render {

        class BSDF {
        private:
            BxDF _bxdf;
            float3 _ng;
            Frame _shading_frame;
            MAKE_SOA_FRIEND(BSDF);
        public:
            XPU BSDF() = default;

            XPU BSDF(float3 ng, float3 ns, float3 dp_dus, BxDF bxdf)
                    : _ng(ng), _shading_frame(Frame::from_xz(dp_dus, ns)), _bxdf(std::move(bxdf)) {}

            NDSC_XPU Spectrum eval(float3 wo_world, float3 wi_world,
                                 TransportMode mode = TransportMode::Radiance) const;

            NDSC_XPU float PDF(float3 wo_world, float3 wi_world,
                               TransportMode mode = TransportMode::Radiance,
                               BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const;

            NDSC_XPU lstd::optional<BSDFSample> sample_f(float3 world_wo, float uc, float2 u,
                                                         TransportMode mode = TransportMode::Radiance,
                                                         BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const;

            NDSC_XPU float4 base_color() const;

            NDSC_XPU Spectrum rho_hd(float3 wo_world, BufferView<const float> uc,
                                   BufferView<const float2> u2) const;

            NDSC_XPU Spectrum rho_hh(BufferView<const float2> u1, BufferView<const float> uc,
                                   BufferView<const float2> u2) const;

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