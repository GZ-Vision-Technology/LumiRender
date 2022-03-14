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
#include "core/memory/util.h"

namespace luminous {
    inline namespace utility {

        class Image : public Noncopyable, public ImageBase {
        private:
            luminous_fs::path _path;
            std::unique_ptr<const std::byte[]> _pixel;
        public:
            Image() = default;

            Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res, const luminous_fs::path &path);

            Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res);

            Image(Image &&other) noexcept;

            Image(const Image &other) = delete;

            Image &operator=(const Image &other) = delete;

            template<typename T = std::byte>
            const T *pixel_ptr() const { return reinterpret_cast<const T *>(_pixel.get()); }

            template<typename T = std::byte>
            T *pixel_ptr() { return reinterpret_cast<T *>(const_cast<std::byte *>(_pixel.get())); }

            static Image pure_color(float4 color, ColorSpace color_space, uint2 res = make_uint2(1u));

            static Image load(const luminous_fs::path &fn, ColorSpace color_space, float3 scale = make_float3(1.f));

            static Image load_hdr(const luminous_fs::path &fn, ColorSpace color_space, float3 scale = make_float3(1.f));

            static Image load_exr(const luminous_fs::path &fn, ColorSpace color_space, float3 scale = make_float3(1.f));

            static Image create_empty(PixelFormat pixel_format, uint2 res) {
                size_t size_in_bytes = pixel_size(pixel_format) * res.x * res.y;
                auto pixel = new_array(size_in_bytes);
                return {pixel_format, pixel, res};
            }

            template<typename T>
            static Image from_data(T *data, uint2 res) {
                size_t size_in_bytes = sizeof(T) * res.x * res.y;
                auto pixel = new_array(size_in_bytes);
                auto pixel_format = PixelFormatImpl<T>::format;
                std::memcpy(pixel, data, size_in_bytes);
                return {pixel_format, pixel, res};
            }

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

            void convert_to_8bit_image();

            void convert_to_32bit_image();

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