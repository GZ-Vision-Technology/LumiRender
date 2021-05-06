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

        class RGBSpectrum : public float3 {
        public:
            using scalar_t = float;
            using vector_t = float3;

            XPU explicit RGBSpectrum(scalar_t r = 0, scalar_t g = 0, scalar_t b = 0)
                    : vector_t(r, g, b) {}

            XPU explicit RGBSpectrum(vector_t vec)
                    : vector_t(vec) {}

            XPU [[nodiscard]] scalar_t R() const noexcept { return x; }

            XPU [[nodiscard]] scalar_t G() const noexcept { return y; }

            XPU [[nodiscard]] scalar_t B() const noexcept { return z; }

            XPU [[nodiscard]] vector_t vec() const noexcept {
                return make_float3(x,y,z);
            }

            XPU [[nodiscard]] scalar_t X() const noexcept {
                return dot(*this, vector_t(0.412453f, 0.357580f, 0.180423f));
            }

            XPU [[nodiscard]] scalar_t Y() const noexcept {
                return dot(*this, vector_t(0.212671f, 0.715160f, 0.072169f));
            }

            XPU [[nodiscard]] scalar_t Z() const noexcept {
                return dot(*this, vector_t(0.019334f, 0.119193f, 0.950227f));
            }

            XPU [[nodiscard]] scalar_t luminance() const noexcept {
                return Y();
            };

            XPU [[nodiscard]] vector_t XYZ() const noexcept {
                return vector_t(X(), Y(), Z());
            }

            XPU [[nodiscard]] scalar_t average() const noexcept {
                return dot(*this, vector_t(1.f, 1.f, 1.f)) / 3;
            }

            template<typename T>
            XPU static T linear_to_srgb(T L) {
                return select((L < T(0.0031308f)),
                              (L * T(12.92f)),
                              (T(1.055f) * pow(L, T(1.0f / 2.4f)) - T(0.055f)));
            }

            XPU static RGBSpectrum linear_to_srgb(RGBSpectrum color) {
                auto vec = (vector_t) (color);
                return RGBSpectrum(linear_to_srgb(vec));
            }

            template<typename T>
            XPU static T srgb_to_linear(T S) {
                return select((S < T(0.04045f)),
                              (S / T(12.92f)),
                              (pow((S + 0.055f) * 1.f / 1.055f, (float) 2.4f)));
            }

            XPU static RGBSpectrum srgb_to_linear(RGBSpectrum color) {
                auto vec = (vector_t) (color);
                return RGBSpectrum(srgb_to_linear(vec));
            }

            XPU [[nodiscard]] bool is_black() const noexcept {
                return is_zero(*this);
            }

        };

        inline XPU uint32_t make_8bit(const float f) {
            return min(255, max(0, int(f * 256.f)));
        }

        inline XPU uint32_t make_rgba(const float3 color) {
            return (make_8bit(color.x) << 0) +
                   (make_8bit(color.y) << 8) +
                   (make_8bit(color.z) << 16) +
                   (0xffU << 24);
        }

        inline XPU uint32_t make_rgba(const float4 color) {
            return (make_8bit(color.x) << 0) +
                   (make_8bit(color.y) << 8) +
                   (make_8bit(color.z) << 16) +
                   (make_8bit(color.w) << 24);
        }

        using Spectrum = RGBSpectrum;
    }
}