//
// Created by Zero on 2021/4/29.
//


#pragma once

#include "graphics/math/common.h"
#include "graphics/sampling/warp.h"
#include "graphics/geometry/common.h"
#include "graphics/lstd/lstd.h"
#include "graphics/optics/rgb.h"

namespace luminous {
    inline namespace render {
        // BxDFFlags Definition
        enum BxDFFlags {
            Unset = 0,
            Reflection = 1 << 0,
            Transmission = 1 << 1,
            Diffuse = 1 << 2,
            Glossy = 1 << 3,
            Specular = 1 << 4,
            // Composite _BxDFFlags_ definitions
            DiffuseReflection = Diffuse | Reflection,
            DiffuseTransmission = Diffuse | Transmission,
            GlossyReflection = Glossy | Reflection,
            GlossyTransmission = Glossy | Transmission,
            SpecularReflection = Specular | Reflection,
            SpecularTransmission = Specular | Transmission,
            All = Diffuse | Glossy | Specular | Reflection | Transmission
        };

        enum class TransportMode {
            Radiance,
            Importance
        };

        enum class BxDFReflTransFlags {
            Unset = 0,
            Reflection = 1 << 0,
            Transmission = 1 << 1,
            All = Reflection | Transmission
        };

        NDSC_XPU_INLINE BxDFFlags operator|(BxDFFlags a, BxDFFlags b) {
            return BxDFFlags((int) a | (int) b);
        }

        NDSC_XPU_INLINE int operator&(BxDFFlags a, BxDFFlags b) {
            return ((int) a & (int) b);
        }

        NDSC_XPU_INLINE int operator&(BxDFFlags a, BxDFReflTransFlags b) {
            return ((int) a & (int) b);
        }

        NDSC_XPU_INLINE int operator&(BxDFReflTransFlags a, BxDFFlags b) {
            return ((int) a & (int) b);
        }

        NDSC_XPU_INLINE int operator&(BxDFReflTransFlags a, BxDFReflTransFlags b) {
            return ((int) a & (int) b);
        }

        NDSC_XPU_INLINE BxDFFlags &operator|=(BxDFFlags &a, BxDFFlags b) {
            (int &) a |= int(b);
            return a;
        }

        NDSC_XPU_INLINE bool is_reflective(BxDFFlags f) {
            return f & BxDFFlags::Reflection;
        }

        NDSC_XPU_INLINE bool is_transmissive(BxDFFlags f) {
            return f & BxDFFlags::Transmission;
        }

        NDSC_XPU_INLINE bool is_diffuse(BxDFFlags f) {
            return f & BxDFFlags::Diffuse;
        }

        NDSC_XPU_INLINE bool is_glossy(BxDFFlags f) {
            return f & BxDFFlags::Glossy;
        }

        NDSC_XPU_INLINE bool is_specular(BxDFFlags f) {
            return f & BxDFFlags::Specular;
        }

        NDSC_XPU_INLINE bool is_non_specular(BxDFFlags f) {
            return (f & (BxDFFlags::Diffuse | BxDFFlags::Glossy));
        }

        struct BSDFSample {
            Spectrum f_val;
            float3 wi;
            float PDF;
            BxDFFlags flags;
            float eta = 1;

            NDSC_XPU BSDFSample() = default;

            NDSC_XPU BSDFSample(const Spectrum &val, float3 wi_, float PDF_, BxDFFlags flags_, float eta_ = 1)
                    : f_val(val), wi(wi_), PDF(PDF_), flags(flags_), eta(eta_) {}

            NDSC_XPU bool is_non_specular() const {
                return luminous::is_non_specular(flags);
            }

            NDSC_XPU bool is_reflective() const {
                return luminous::is_reflective(flags);
            }

            NDSC_XPU bool is_transmissive() const {
                return luminous::is_transmissive(flags);
            }

            NDSC_XPU bool is_diffuse() const {
                return luminous::is_diffuse(flags);
            }

            NDSC_XPU bool is_glossy() const {
                return luminous::is_glossy(flags);
            }

            NDSC_XPU bool is_specular() const {
                return luminous::is_specular(flags);
            }
        };
    }
}