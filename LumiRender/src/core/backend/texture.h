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

    class DTexture {
    public:
        class Impl : public ImageBase {
        public:
            Impl(PixelFormat pixel_format, uint2 resolution)
                    : ImageBase(pixel_format, resolution) {}

            virtual void copy_from(Dispatcher &dispatcher, const Image &image) = 0;

            NDSC virtual Image download() const = 0;

            virtual void copy_from(const Image &image) = 0;

            NDSC virtual uint64_t tex_handle() const = 0;

            virtual ~Impl() = default;
        };

        explicit DTexture(std::unique_ptr<Impl> impl)
                : _impl(move(impl)) {}

        NDSC uint32_t width() const { return _impl->width(); }

        NDSC uint32_t height() const { return _impl->height(); }

        NDSC uint2 resolution() const { return _impl->resolution(); }

        NDSC size_t pixel_num() const { return _impl->pixel_num(); }

        NDSC PixelFormat pixel_format() const { return _impl->pixel_format(); }

        void copy_from(Dispatcher &dispatcher, const Image &image) {
            _impl->copy_from(dispatcher, image);
        }

        void copy_from(const Image &image) {
            _impl->copy_from(image);
        }

        NDSC Image download() const {
            return std::move(_impl->download());
        }

        template<typename T = uint64_t>
        NDSC T tex_handle() const {
            return (T) _impl->tex_handle();
        }

    private:
        std::unique_ptr<Impl> _impl;

    };
}