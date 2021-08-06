//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/concepts.h"
#include "base_libs/math/common.h"
#include "dispatcher.h"
#include "util/image.h"
#include "buffer.h"

namespace luminous {

    class DeviceTexture {
    public:
        class Impl : public ImageBase {
        public:
            Impl(PixelFormat pixel_format, uint2 resolution)
                    : ImageBase(pixel_format, resolution) {}

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

        PixelFormat pixel_format() const { return _impl->pixel_format(); }

        void copy_from(Dispatcher &dispatcher, const Image &image) {
            _impl->copy_from(dispatcher, image);
        }

        void copy_from(const Image &image) {
            _impl->copy_from(image);
        }

        Image download() const {
            return std::move(_impl->download());
        }

        template<typename T = void *>
        T tex_handle() const {
            return (T) _impl->tex_handle();
        }

    private:
        std::unique_ptr<Impl> _impl;

    };
}