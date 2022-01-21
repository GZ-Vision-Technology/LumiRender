//
// Created by Zero on 2021/2/9.
//


#pragma once

#include "../math/common.h"

namespace luminous {
    inline namespace optics {

        enum ColorSpace {
            LINEAR,
            SRGB
        };

        ND_XPU_INLINE float luminance(float3 v) {
            return dot(v, make_float3(0.212671f, 0.715160f, 0.072169f));
        }

        ND_XPU_INLINE float luminance(float4 v) {
            return dot(make_float3(v), make_float3(0.212671f, 0.715160f, 0.072169f));
        }

        class RGBSpectrum : public float3 {
        public:
            static constexpr int nSamples = 3;

            using scalar_t = float;
            using vector_t = float3;

            LM_XPU RGBSpectrum(scalar_t r, scalar_t g, scalar_t b)
                    : vector_t(r, g, b) {}

            LM_XPU RGBSpectrum(scalar_t c = 0.f)
                    : vector_t(c, c, c) {}

            LM_XPU RGBSpectrum(vector_t vec)
                    : vector_t(vec) {}

            LM_XPU RGBSpectrum(float4 vec)
                    : RGBSpectrum(make_float3(vec)) {}

            LM_XPU const scalar_t& operator[](int i) const {
                DCHECK_LE(i, nSamples);
                return (&x)[i];
            }

            LM_XPU scalar_t& operator[](int i) {
                DCHECK_LE(i, nSamples);
                return (&x)[i];
            }

            ND_XPU_INLINE scalar_t R() const noexcept { return x; }

            ND_XPU_INLINE scalar_t G() const noexcept { return y; }

            ND_XPU_INLINE scalar_t B() const noexcept { return z; }

            ND_XPU_INLINE vector_t vec() const noexcept {
                return make_float3(x, y, z);
            }

            ND_XPU_INLINE scalar_t X() const noexcept {
                return dot(*this, vector_t(0.412453f, 0.357580f, 0.180423f));
            }

            ND_XPU_INLINE scalar_t Y() const noexcept {
                return dot(*this, vector_t(0.212671f, 0.715160f, 0.072169f));
            }

            ND_XPU_INLINE scalar_t Z() const noexcept {
                return dot(*this, vector_t(0.019334f, 0.119193f, 0.950227f));
            }

            ND_XPU_INLINE scalar_t luminance() const noexcept {
                return Y();
            };

            ND_XPU_INLINE scalar_t max_comp() const noexcept {
                return max({x, y, z});
            }

            ND_XPU_INLINE vector_t XYZ() const noexcept {
                return vector_t(X(), Y(), Z());
            }

            ND_XPU_INLINE scalar_t average() const noexcept {
                return dot(*this, vector_t(1.f, 1.f, 1.f)) / 3;
            }

            template<typename T>
            LM_ND_XPU static T linear_to_srgb(T L) {
                return select((L < T(0.0031308f)),
                              (L * T(12.92f)),
                              (T(1.055f) * pow(L, T(1.0f / 2.4f)) - T(0.055f)));
            }

            LM_ND_XPU static RGBSpectrum linear_to_srgb(RGBSpectrum color) {
                auto vec = (vector_t) (color);
                return RGBSpectrum(linear_to_srgb(vec));
            }

            template<typename T>
            LM_ND_XPU static T srgb_to_linear(T S) {
                return select((S < T(0.04045f)),
                              (S / T(12.92f)),
                              (pow((S + 0.055f) * 1.f / 1.055f, (float) 2.4f)));
            }

            LM_ND_XPU static RGBSpectrum srgb_to_linear(RGBSpectrum color) {
                auto vec = (vector_t) (color);
                return RGBSpectrum(srgb_to_linear(vec));
            }

            ND_XPU_INLINE bool is_black() const noexcept {
                return is_zero(*this);
            }

            ND_XPU_INLINE bool not_black() const noexcept {
                return nonzero(*this);
            }

        };

        ND_XPU_INLINE uint32_t make_8bit(const float f) {
            return min(255, max(0, int(f * 256.f)));
        }

        ND_XPU_INLINE uint32_t make_rgba(const float3 color) {
            return (make_8bit(color.x) << 0) +
                   (make_8bit(color.y) << 8) +
                   (make_8bit(color.z) << 16) +
                   (0xffU << 24);
        }

        ND_XPU_INLINE uint32_t make_rgba(const float4 color) {
            return (make_8bit(color.x) << 0) +
                   (make_8bit(color.y) << 8) +
                   (make_8bit(color.z) << 16) +
                   (make_8bit(color.w) << 24);
        }

        using Spectrum = RGBSpectrum;
    }
}