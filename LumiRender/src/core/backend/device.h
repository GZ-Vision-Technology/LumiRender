//
// Created by Zero on 2021/2/17.
//


#pragma once

#include "dispatcher.h"
#include "buffer.h"
#include "core/context.h"

namespace luminous {

    class Device : public Noncopyable {
    public:
        class Impl {
        public:
            virtual RawBuffer allocate_buffer(size_t bytes) = 0;

            virtual Dispatcher new_dispatcher() = 0;

            virtual ~Impl() = default;
        };

        template<typename T>
        Buffer <T> allocate_buffer(size_t n_elements) {
            return Buffer<T>(_impl->allocate_buffer(n_elements * sizeof(T)));
        }

        Dispatcher new_dispatcher() { return _impl->new_dispatcher(); }

        Device(std::unique_ptr<Impl> impl) : _impl(std::move(impl)) {}

    protected:
        std::unique_ptr<Impl> _impl;
    };
}