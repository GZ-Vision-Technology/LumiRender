//
// Created by Zero on 2021/2/17.
//


#pragma once

#include "dispatcher.h"
#include "buffer.h"
#include "texture.h"
#include "core/context.h"

namespace luminous {

    class Device : public Noncopyable {
    public:
        class Impl {
        public:
            virtual RawBuffer allocate_buffer(size_t bytes) = 0;

            virtual DeviceTexture allocate_texture(PixelFormat pixel_format, uint2 resolution) = 0;

            virtual Dispatcher new_dispatcher() = 0;

            virtual ~Impl() = default;
        };

        template<typename T = std::byte>
        Buffer<T> allocate_buffer(size_t n_elements) {
            return Buffer<T>(_impl->allocate_buffer(n_elements * sizeof(T)));
        }

        DeviceTexture allocate_texture(PixelFormat pixel_format, uint2 resolution) {
            return _impl->allocate_texture(pixel_format, resolution);
        }

        Dispatcher new_dispatcher() { return _impl->new_dispatcher(); }



        explicit Device(std::unique_ptr<Impl> impl) : _impl(std::move(impl)) {}

    protected:
        std::unique_ptr<Impl> _impl;
    };
}