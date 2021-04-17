//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/concepts.h"
#include "graphics/math/common.h"
#include "dispatcher.h"
#include "util/image.h"

namespace luminous {



    class DeviceTexture {
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

        DeviceTexture(std::unique_ptr<Impl> impl)
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