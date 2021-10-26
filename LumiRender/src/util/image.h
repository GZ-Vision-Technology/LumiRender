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
        public:
            Image() = default;

            Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res, const luminous_fs::path &path);

            Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res);

            Image(Image &&other) noexcept;

            template<typename T = std::byte>
            const T *pixel_ptr() const { return reinterpret_cast<const T *>(_pixel.get()); }

            LM_NODISCARD bool is_8bit() const;

            LM_NODISCARD bool is_32bit() const;

            static Image load(const luminous_fs::path &fn, ColorSpace color_space);

            static Image load_hdr(const luminous_fs::path &fn, ColorSpace color_space);

            static Image load_exr(const luminous_fs::path &fn, ColorSpace color_space);

            /**
             * ".bmp" or ".png" or ".tga" or ".jpg" or ".jpeg"
             */
            static Image load_other(const luminous_fs::path &fn, ColorSpace color_space);

            void convert_to_32bit();

            void convert_to_8bit();

            template<typename Func>
            void for_each_pixel(Func func) const {
                auto p = _pixel.get();
                int stride = pixel_size(_pixel_format);
                parallel_for(pixel_num(), [&](uint i, uint tid){
                    const std::byte *pixel = p + stride * i;
                    func(pixel, i);
                });
            }

            template<typename Func>
            void for_each_pixel(Func func) {
                auto p = _pixel.get();
                int stride = pixel_size(_pixel_format);
                parallel_for(pixel_num(), [&](uint i, uint tid){
                    std::byte *pixel = const_cast<std::byte*>(p + stride * i);
                    func(pixel, i);
                });
            }

            void save_image(const luminous_fs::path &fn);

            void save_hdr(const luminous_fs::path &fn);

            void save_exr(const luminous_fs::path &fn);

            /**
             * ".bmp" or ".png" or ".tga" or ".jpg" or ".jpeg"
             */
            void save_other(const luminous_fs::path &fn);
        };
    } // luminous::utility
} // luminous

#endif