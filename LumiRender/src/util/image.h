//
// Created by Zero on 2021/4/16.
//


#pragma once

#include "graphics/math/common.h"
#include "graphics/optics/rgb.h"
#include "core/concepts.h"
#include "pixel_format.h"
#include <utility>



namespace luminous {
    inline namespace utility {
        using namespace std;

        class Image : public Noncopyable {
        private:
            PixelFormat _pixel_format;
            uint2 _resolution;
            std::filesystem::path _path;
            std::unique_ptr<const std::byte[]> _pixel;
        public:
            Image() = default;

            Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res, const std::filesystem::path& path);

            Image(Image &&other) noexcept;

            uint2 resolution() const;

            size_t pixel_num() const;

            bool is_8bit() const;

            bool is_32bit() const;

            PixelFormat pixel_format() const;

            int channels() const;

            static Image load(const filesystem::path &fn, ColorSpace color_space);

            static Image load_hdr(const filesystem::path &fn, ColorSpace color_space);

            static Image load_exr(const filesystem::path &fn, ColorSpace color_space);

            /**
             * ".bmp" or ".png" or ".tga" or ".jpg" or ".jpeg"
             */
            static Image load_other(const filesystem::path &fn, ColorSpace color_space);

            void convert_to_32bit();

            void convert_to_8bit();

            void save_image(const filesystem::path &fn);

            void save_hdr(const filesystem::path &fn);

            void save_exr(const filesystem::path &fn);

            /**
             * ".bmp" or ".png" or ".tga" or ".jpg" or ".jpeg"
             */
            void save_other(const filesystem::path &fn);
        };
    } // luminous::utility
} // luminous