//
// Created by Zero on 2021/4/18.
//


#pragma once

#include "graphics/math/common.h"

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

        NDSC_XPU_INLINE size_t pixel_size(PixelFormat pixel_format) {
            switch (pixel_format) {
                case PixelFormat::R8U:
                    return sizeof(uchar);
                case PixelFormat::RG8U:
                    return sizeof(uchar2);
                case PixelFormat::RGBA8U:
                    return sizeof(uchar4);
                case PixelFormat::R32F:
                    return sizeof(float);
                case PixelFormat::RG32F:
                    return sizeof(float2);
                case PixelFormat::RGBA32F:
                    return sizeof(float4);
            }
        }

        NDSC_XPU_INLINE int channel_num(PixelFormat pixel_format) {
            if (pixel_format == PixelFormat::R8U || pixel_format == PixelFormat::R32F) { return 1u; }
            if (pixel_format == PixelFormat::RG8U || pixel_format == PixelFormat::RG32F) { return 2u; }
            return 4u;
        }

        template<typename T>
        using PixelFormatImpl = detail::PixelFormatImpl<T>;
    }
}