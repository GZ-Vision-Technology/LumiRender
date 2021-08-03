//
// Created by Zero on 2021/4/16.
//


#pragma once


#ifndef __CUDACC__

// cpu only
#include "graphics/math/common.h"
#include "graphics/optics/rgb.h"
#include "core/concepts.h"
#include "image_base.h"
#include <utility>

namespace luminous {
    inline namespace utility {

        class Image : public Noncopyable, public ImageBase {
        private:
            std::filesystem::path _path;
            std::unique_ptr<const std::byte[]> _pixel;
        public:
            Image() = default;

            Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res, const std::filesystem::path &path);

            Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res);

            Image(Image &&other) noexcept;

            template<typename T = std::byte>
            const T *pixel_ptr() const { return reinterpret_cast<const T *>(_pixel.get()); }

            NDSC bool is_8bit() const;

            NDSC bool is_32bit() const;

            static Image load(const std::filesystem::path &fn, ColorSpace color_space);

            static Image load_hdr(const std::filesystem::path &fn, ColorSpace color_space);

            static Image load_exr(const std::filesystem::path &fn, ColorSpace color_space);

            /**
             * ".bmp" or ".png" or ".tga" or ".jpg" or ".jpeg"
             */
            static Image load_other(const std::filesystem::path &fn, ColorSpace color_space);

            void convert_to_32bit();

            void convert_to_8bit();

            template<typename Func>
            void for_each_pixel(Func func) const {
                auto p = _pixel.get();
                int stride = pixel_size(_pixel_format);
                for (int i = 0; i < pixel_num(); ++i) {
                    p += stride;
                    func(p, i);
                }
            }

            void save_image(const std::filesystem::path &fn);

            void save_hdr(const std::filesystem::path &fn);

            void save_exr(const std::filesystem::path &fn);

            /**
             * ".bmp" or ".png" or ".tga" or ".jpg" or ".jpeg"
             */
            void save_other(const std::filesystem::path &fn);
        };
    } // luminous::utility
} // luminous

#endif