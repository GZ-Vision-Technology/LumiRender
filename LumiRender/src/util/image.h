//
// Created by Zero on 2021/4/16.
//


#pragma once

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

            bool is_8bit() const;

            bool is_32bit() const;

            static Image load(const std::filesystem::path &fn, ColorSpace color_space);

            static Image load_hdr(const std::filesystem::path &fn, ColorSpace color_space);

            static Image load_exr(const std::filesystem::path &fn, ColorSpace color_space);

            /**
             * ".bmp" or ".png" or ".tga" or ".jpg" or ".jpeg"
             */
            static Image load_other(const std::filesystem::path &fn, ColorSpace color_space);

            void convert_to_32bit();

            void convert_to_8bit();

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