//
// Created by Zero on 2021/5/1.
//


#pragma once

#include "render/scattering/bxdf.h"
#include "base_libs/optics/rgb.h"
#include "render/integrators/wavefront/soa.h"
#include "bsdfs.h"

namespace luminous {
    inline namespace render {

        class BSDFWrapper {
        private:
            BSDF _bsdf;
            float3 _ng;
            Frame _shading_frame;
        public:
            LM_XPU BSDFWrapper() = default;

            LM_XPU BSDFWrapper(float3 ng, float3 ns, float3 dp_dus, BSDF bsdf)
                    : _ng(ng), _shading_frame(Frame::from_xz(dp_dus, ns)), _bsdf(std::move(bsdf)) {}

            LM_ND_XPU Spectrum eval(float3 wo_world, float3 wi_world,
                                    BxDFFlags sample_flags = BxDFFlags::All,
                                    TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU float PDF(float3 wo_world, float3 wi_world,
                                BxDFFlags sample_flags = BxDFFlags::All,
                                TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU BSDFSample sample_f(float3 world_wo, float uc, float2 u,
                                          BxDFFlags sample_flags = BxDFFlags::All,
                                          TransportMode mode = TransportMode::Radiance) const;

            LM_ND_XPU Spectrum color() const;

            LM_ND_XPU float3 to_local(float3 val) const {
                return _shading_frame.to_local(val);
            }

            LM_ND_XPU float3 to_world(float3 val) const {
                return _shading_frame.to_world(val);
            }

            LM_ND_XPU bool is_non_specular() const {
                return luminous::is_non_specular(_bsdf.flags());
            }

            LM_ND_XPU bool is_reflective() const {
                return luminous::is_reflective(_bsdf.flags());
            }

            LM_ND_XPU bool is_transmissive() const {
                return luminous::is_transmissive(_bsdf.flags());
            }

            LM_ND_XPU bool is_diffuse() const {
                return luminous::is_diffuse(_bsdf.flags());
            }

            LM_ND_XPU bool is_glossy() const {
                return luminous::is_glossy(_bsdf.flags());
            }

            LM_ND_XPU bool is_specular() const {
                return luminous::is_specular(_bsdf.flags());
            }
        };

    } // luminous::render
} // luminous