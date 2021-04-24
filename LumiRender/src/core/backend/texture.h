//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/concepts.h"
#include "graphics/math/common.h"
#include "dispatcher.h"
#include "util/image.h"
#include "buffer.h"

namespace luminous {

    class DeviceTexture {
    public:
        class Impl {
        protected:
            PixelFormat _pixel_format;
            uint2 _resolution;
        public:
            Impl(PixelFormat pixel_format, uint2 resolution)
                    : _pixel_format(pixel_format), _resolution(resolution) {}

            size_t pitch_byte_size() const { return _resolution.x * pixel_size(_pixel_format); }

            uint32_t width() const { return _resolution.x; }

            uint32_t height() const { return _resolution.y; }

            PixelFormat format() const { return _pixel_format; }

            uint2 resolution() const { return _resolution; }

            size_t pixel_num() const { return width() * height(); }

            size_t size_in_bytes() const { return pixel_size(_pixel_format) * pixel_num() * channels(); }

            int channels() const { return channel_num(_pixel_format); }

            virtual void copy_from(Dispatcher &dispatcher, const Image &image) = 0;

            virtual Image download() const = 0;

            virtual void copy_from(const Image &image) = 0;

            virtual void *tex_handle() const = 0;

            virtual ~Impl() = default;
        };

        DeviceTexture(std::unique_ptr<Impl> impl)
                : _impl(move(impl)) {}

        uint32_t width() const { return _impl->width(); }

        uint32_t height() const { return _impl->height(); }

        uint2 resolution() const { return _impl->resolution(); }

        size_t pixel_num() const { return _impl->pixel_num(); }

        void copy_from(Dispatcher &dispatcher, const Image &image) {
            _impl->copy_from(dispatcher, image);
        }

        void copy_from(const Image &image) {
            _impl->copy_from(image);
        }

        Image download() const {
            return move(_impl->download());
        }

        template<typename T = void *>
        T tex_handle() const {
            return (T) _impl->tex_handle();
        }

    private:
        std::unique_ptr<Impl> _impl;

    };
}