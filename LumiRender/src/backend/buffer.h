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
            virtual void download(Dispatcher &dispatcher, size_t offset, size_t size, void *host_data) = 0;

            virtual void upload(Dispatcher &dispatcher, size_t offset, size_t size, const void *host_data) = 0;

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
        using RawBuffer::RawBuffer;

        Buffer(RawBuffer buf) : RawBuffer(std::move(buf)) {}

        T *data() const { return reinterpret_cast<T *>(ptr()); }

        size_t size() const { return impl->size() / sizeof(T); }

        void download(Dispatcher &dispatcher, size_t offset, size_t size, T *host_data) {
            assert(offset * sizeof(T) + size * sizeof(T) <= impl->size());
            impl->download(dispatcher, offset * sizeof(T), size * sizeof(T), host_data);
        }

        void upload(Dispatcher &dispatcher, size_t offset, size_t size, const T *host_data) {
            assert(offset * sizeof(T) + size * sizeof(T) <= impl->size());
            impl->upload(dispatcher, offset * sizeof(T), size * sizeof(T), host_data);
        }
    };
}