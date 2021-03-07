//
// Created by Zero on 2021/2/17.
//


#pragma once

#include "core/concepts.h"
#include <iostream>
#include <limits>
#include <functional>
#include <utility>
#include "dispatcher.h"

namespace luminous {

    class Dispatcher;

    class RawBuffer {
    public:
        class Impl {
        public:
            virtual void download(Dispatcher &dispatcher, void *host_data, size_t size, size_t offset = 0) = 0;

            virtual void upload(Dispatcher &dispatcher, const void *host_data, size_t size, size_t offset = 0) = 0;

            virtual size_t size() const = 0;

            virtual void *ptr() = 0;

            virtual ~Impl() = default;
        };

        RawBuffer(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {}

        Impl *impl_mut() const { return impl.get(); }

        void *ptr() const { return impl->ptr(); }

    protected:
        std::unique_ptr<Impl> impl;
    };

    template<class T>
    class Buffer : public RawBuffer {
    public:
        using value_type = T;

        using RawBuffer::RawBuffer;

        Buffer(RawBuffer buf) : RawBuffer(std::move(buf)) {}

        T *data() const { return reinterpret_cast<T *>(ptr()); }

        size_t size() const { return impl->size() / sizeof(T); }

        void download(Dispatcher &dispatcher, T *host_data, size_t size, size_t offset = 0) {
            assert(offset * sizeof(T) + size * sizeof(T) <= impl->size());
            impl->download(dispatcher, host_data, size * sizeof(T), offset * sizeof(T));
        }

        void upload(Dispatcher &dispatcher, const T *host_data, size_t size, size_t offset = 0) {
            assert(offset * sizeof(T) + size * sizeof(T) <= impl->size());
            impl->upload(dispatcher, host_data, size * sizeof(T), offset * sizeof(T));
        }
    };
}