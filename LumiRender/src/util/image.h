//
// Created by Zero on 2021/4/16.
//


#pragma once

#include "graphics/math/common.h"
#include "graphics/optics/rgb.h"
#include "core/concepts.h"
#include <utility>



namespace luminous {
    inline namespace utility {
        using namespace std;
        enum struct PixelFormat : uint32_t {
            R8U, RG8U, RGBA8U,
            R32F, RG32F, RGBA32F,
        };

        namespace detail {

            template<typename T>
            struct PixelFormatImpl {

                template<typename U>
                static constexpr auto always_false = false;

                static_assert(always_false<T>, "Unsupported type for pixel format.");
            };

#define MAKE_PIXEL_FORMAT_OF_TYPE(Type, f)                  \
        template<>                                          \
        struct PixelFormatImpl<Type> {                      \
            static constexpr auto format = PixelFormat::f;  \
            static constexpr auto pixel_size = sizeof(Type);\
        };                                                  \

            MAKE_PIXEL_FORMAT_OF_TYPE(uchar, R8U)
            MAKE_PIXEL_FORMAT_OF_TYPE(uchar2, RG8U)
            MAKE_PIXEL_FORMAT_OF_TYPE(uchar4, RGBA8U)
            MAKE_PIXEL_FORMAT_OF_TYPE(float, R32F)
            MAKE_PIXEL_FORMAT_OF_TYPE(float2, RG32F)
            MAKE_PIXEL_FORMAT_OF_TYPE(float4, RGBA32F)

#undef MAKE_PIXEL_FORMAT_OF_TYPE
        }

        class Image : public Noncopyable {
        private:
            PixelFormat _pixel_format;
            uint2 _resolution;
            std::unique_ptr<const std::byte[]> _pixel;
        public:
            Image() = default;

            Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res);

            Image(Image &&other) noexcept;

            uint2 resolution() const;

            PixelFormat pixel_format() const;

            static Image load(const filesystem::path &fn, ColorSpace color_space);

            static Image load_hdr(const filesystem::path &fn, ColorSpace color_space);

            static Image load_exr(const filesystem::path &fn, ColorSpace color_space);

            /**
             * ".bmp" or ".png" or ".tga" or ".jpg" or ".jpeg"
             */
            static Image load_other(const filesystem::path &fn, ColorSpace color_space);

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