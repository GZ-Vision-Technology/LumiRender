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
            virtual void download(void *host_data, size_t size = 0, size_t offset = 0) = 0;

            virtual void download_async(Dispatcher &dispatcher, void *host_data, size_t size = 0, size_t offset = 0) = 0;

            virtual void upload(const void *host_data, size_t size = 0, size_t offset = 0) = 0;

            virtual void upload_async(Dispatcher &dispatcher, const void *host_data, size_t size = 0, size_t offset = 0) = 0;

            virtual void *address(size_t offset = 0) const = 0;

            virtual size_t size() const = 0;

            virtual void *ptr() = 0;

            virtual ~Impl() = default;
        };

        RawBuffer(std::unique_ptr<Impl> impl) : _impl(std::move(impl)) {}

        Impl *impl_mut() const { return _impl.get(); }

        template<typename T = void *>
        auto ptr() const { return (T)_impl->ptr(); }

    protected:
        std::unique_ptr<Impl> _impl;
    };

    template<class T>
    class Buffer : public RawBuffer {
    public:
        using value_type = T;

        using RawBuffer::RawBuffer;

        Buffer(RawBuffer buf) : RawBuffer(std::move(buf)) {}

        T *data() const { return reinterpret_cast<T *>(ptr()); }

        template<typename U = void *>
        auto address(size_t offset = 0) const { return (U)_impl->address(offset * sizeof(T)); }

        size_t size() const { return _impl->size() / sizeof(T); }

        void download(T *host_data, size_t size = 0, size_t offset = 0) {
            size = size == 0 ? _impl->size() / sizeof(T) : size;
            assert(offset * sizeof(T) + size * sizeof(T) <= _impl->size());
            _impl->download(host_data, size * sizeof(T), offset * sizeof(T));
        }

        void download_async(Dispatcher &dispatcher, T *host_data, size_t size = 0, size_t offset = 0) {
            size = size == 0 ? _impl->size() / sizeof(T) : size;
            assert(offset * sizeof(T) + size * sizeof(T) <= _impl->size());
            _impl->download_async(dispatcher, host_data, size * sizeof(T), offset * sizeof(T));
        }

        void upload(const T *host_data, size_t size = 0, size_t offset = 0) {
            size = size == 0 ? _impl->size() / sizeof(T) : size;
            assert(offset * sizeof(T) + size * sizeof(T) <= _impl->size());
            _impl->upload(host_data, size * sizeof(T), offset * sizeof(T));
        }

        void upload_async(Dispatcher &dispatcher, const T *host_data, size_t size = 0, size_t offset = 0) {
            size = size == 0 ? _impl->size() / sizeof(T) : size;
            assert(offset * sizeof(T) + size * sizeof(T) <= _impl->size());
            _impl->upload_async(dispatcher, host_data, size * sizeof(T), offset * sizeof(T));
        }
    };
}