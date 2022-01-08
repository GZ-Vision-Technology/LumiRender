//
// Created by Zero on 2021/4/16.
//


#pragma once


#ifndef __CUDACC__

// cpu only
#include "base_libs/math/common.h"
#include "base_libs/optics/rgb.h"
#include "core/concepts.h"
#include "image_base.h"
#include "parallel.h"

namespace luminous {
    inline namespace utility {

        class Image : public Noncopyable, public ImageBase {
        private:
            luminous_fs::path _path;
            std::unique_ptr<const std::byte[]> _pixel;
        private:
            void _convert_to_32bit();

            void _convert_to_8bit();

            void _save_hdr(const luminous_fs::path &fn);

            void _save_exr(const luminous_fs::path &fn);

            /**
             * ".bmp" or ".png" or ".tga" or ".jpg" or ".jpeg"
             */
            void _save_other(const luminous_fs::path &fn);

        public:
            Image() = default;

            Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res, const luminous_fs::path &path);

            Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res);

            Image(Image &&other) noexcept;

            template<typename T = std::byte>
            const T *pixel_ptr() const { return reinterpret_cast<const T *>(_pixel.get()); }

            LM_NODISCARD bool is_8bit_image() const { return is_8bit(_pixel_format); }

            LM_NODISCARD bool is_32bit_image() const { return is_32bit(_pixel_format); }

            static Image pure_color(float4 color, ColorSpace color_space);

            static Image load(const luminous_fs::path &fn, ColorSpace color_space, float3 scale = make_float3(1.f));

            static Image load_hdr(const luminous_fs::path &fn, ColorSpace color_space, float3 scale = make_float3(1.f));

            static Image load_exr(const luminous_fs::path &fn, ColorSpace color_space, float3 scale = make_float3(1.f));

            /**
             * ".bmp" or ".png" or ".tga" or ".jpg" or ".jpeg"
             */
            static Image load_other(const luminous_fs::path &fn, ColorSpace color_space,
                                    float3 scale = make_float3(1.f));

            template<typename Func>
            void for_each_pixel(Func func) const {
                auto p = _pixel.get();
                int stride = pixel_size(_pixel_format);
                parallel_for(pixel_num(), [&](uint i, uint tid) {
                    const std::byte *pixel = p + stride * i;
                    func(pixel, i);
                });
            }

            template<typename Func>
            void for_each_pixel(Func func) {
                auto p = _pixel.get();
                int stride = pixel_size(_pixel_format);
                parallel_for(pixel_num(), [&](uint i, uint tid) {
                    std::byte *pixel = const_cast<std::byte *>(p + stride * i);
                    func(pixel, i);
                });
            }

            void save(const luminous_fs::path &fn);

            static std::pair<PixelFormat, const std::byte *> convert_to_32bit(PixelFormat pixel_format,
                                                                              const std::byte *ptr, uint2 res);

            static std::pair<PixelFormat, const std::byte *> convert_to_8bit(PixelFormat pixel_format,
                                                                             const std::byte *ptr, uint2 res);

            static void save_image(const luminous_fs::path &fn, PixelFormat pixel_format,
                                   uint2 res, const std::byte *ptr);

            static void save_exr(const luminous_fs::path &fn, PixelFormat pixel_format,
                                 uint2 res, const std::byte *ptr);

            static void save_hdr(const luminous_fs::path &fn, PixelFormat pixel_format,
                                 uint2 res, const std::byte *ptr);

            static void save_other(const luminous_fs::path &fn, PixelFormat pixel_format,
                                   uint2 res, const std::byte *ptr);
        };
    } // luminous::utility
} // luminous

#endif