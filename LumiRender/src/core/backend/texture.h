//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/concepts.h"
#include "graphics/math/common.h"
#include "dispatcher.h"

namespace luminous {

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
        };                                                  \

        MAKE_PIXEL_FORMAT_OF_TYPE(uchar, R8U)
        MAKE_PIXEL_FORMAT_OF_TYPE(uchar2, RG8U)
        MAKE_PIXEL_FORMAT_OF_TYPE(uchar4, RGBA8U)
        MAKE_PIXEL_FORMAT_OF_TYPE(float, R32F)
        MAKE_PIXEL_FORMAT_OF_TYPE(float2, RG32F)
        MAKE_PIXEL_FORMAT_OF_TYPE(float4, RGBA32F)

#undef MAKE_PIXEL_FORMAT_OF_TYPE
    }

    class Texture {
    public:
        class Impl {
        public:
            virtual uint32_t width() const = 0;

            virtual uint32_t height() const = 0;

            virtual PixelFormat format() const = 0;

            virtual void copy_to(Dispatcher &dispatcher, const void *data) const = 0;

            virtual void copy_from(Dispatcher &dispatcher, const void *data) const = 0;

            virtual ~Impl() = default;
        };

        Texture(std::unique_ptr<Impl> impl)
                : _impl(move(impl)) {}

        uint32_t width() const { return _impl->width(); };

        uint32_t height() const { return _impl->height(); };

        void copy_to(Dispatcher &dispatcher, const void *data) const {
            _impl->copy_to(dispatcher, data);
        }

        void copy_from(Dispatcher &dispatcher, const void *data) const {
            _impl->copy_from(dispatcher, data);
        }

    private:
        std::unique_ptr<Impl> _impl;

    };
}