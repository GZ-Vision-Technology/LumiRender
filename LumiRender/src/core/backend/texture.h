//
// Created by Zero on 2021/2/18.
//


#pragma once

#include "core/concepts.h"
#include "graphics/math/common.h"
#include "dispatcher.h"

namespace luminous {
    class Texture {
    public:
        class Impl {
        public:
            virtual uint32_t width() const = 0;

            virtual uint32_t height() const = 0;

            virtual void copy_to(Dispatcher &dispatcher, const void *data) const = 0;

            virtual void copy_from(Dispatcher &dispatcher, const void *data) const = 0;

            virtual ~Impl() = default;
        };

        uint32_t width() const { return impl->width(); };

        uint32_t height() const { return impl->height(); };

        void copy_to(Dispatcher &dispatcher, const void *data) const {
            impl->copy_to(dispatcher, data);
        }

        void copy_from(Dispatcher &dispatcher, const void *data) const {
            impl->copy_from(dispatcher, data);
        }
    private:
        std::unique_ptr<Impl> impl;

    };
}