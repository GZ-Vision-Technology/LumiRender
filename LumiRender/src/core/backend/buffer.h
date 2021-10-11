//
// Created by Zero on 2021/2/17.
//


#pragma once

#include "core/concepts.h"
#include <iostream>
#include <limits>
#include <functional>
#include "dispatcher.h"
#include "buffer_view.h"
#include "base_libs/math/common.h"

namespace luminous {

    class Dispatcher;

    class RawBuffer {
    public:
        class Impl {
        public:
            virtual void download(void *host_ptr, size_t size, size_t offset) = 0;

            virtual void download_async(Dispatcher &dispatcher, void *host_ptr, size_t size, size_t offset) = 0;

            virtual void upload(const void *host_ptr, size_t size, size_t offset) = 0;

            virtual void upload_async(Dispatcher &dispatcher, const void *host_ptr,
                                      size_t size, size_t offset) = 0;

            virtual void memset(uint32_t val) = 0;

            LM_NODISCARD virtual void *address(size_t offset) const = 0;

            LM_NODISCARD virtual size_t size() const = 0;

            LM_NODISCARD virtual void *ptr() const = 0;

            virtual ~Impl() = default;
        };

        explicit RawBuffer(std::unique_ptr<Impl> impl) : _impl(std::move(impl)) {}

        LM_NODISCARD Impl *impl_mut() const {
            DCHECK(valid());
            return _impl.get();
        }

        template<typename U = void *>
        LM_NODISCARD U ptr() const {
            return (U) (_impl == nullptr ? nullptr : _impl->ptr());
        }

        template<typename U = void *>
        LM_NODISCARD U ptr() {
            return (U) (_impl == nullptr ? nullptr : _impl->ptr());
        }

        LM_NODISCARD bool valid() const {
#ifdef DEBUG_BUILD
            if (_impl == nullptr)
                std::cerr << "invalid buffer !!!" << std::endl;
#endif
            return _impl != nullptr;
        }

        void clear() { _impl.reset(nullptr); }

    protected:
        std::unique_ptr<Impl> _impl;
    };

    template<class T = std::byte>
    class Buffer : public RawBuffer {
    public:
        using value_type = T;

        using RawBuffer::RawBuffer;

        explicit Buffer(RawBuffer buf) : RawBuffer(std::move(buf)) {}

        value_type *data() const { return reinterpret_cast<value_type *>(ptr()); }

        LM_NODISCARD size_t stride_in_bytes() const { return sizeof(value_type); }

        LM_NODISCARD BufferView<value_type> view(size_t offset = 0, size_t count = -1) const {
            count = fix_count(offset, count, size());
            return BufferView<value_type>(data() + offset, count);
        }

        template<typename T>
        LM_NODISCARD Buffer<T> cast() {
            return Buffer<T>(std::move(_impl));
        }

        template<typename U = void *>
        auto address(size_t offset = 0) const {
            DCHECK(valid());
            return (U) _impl->address(offset * sizeof(value_type));
        }

        void memset(uint32_t val = 0) {
            DCHECK(valid());
            _impl->memset(val);
        }

        LM_NODISCARD size_t size() const {
            return size_in_bytes() / sizeof(value_type);
        }

        LM_NODISCARD size_t size_in_bytes() const {
            return _impl == nullptr ? 0 : _impl->size();
        }

        LM_NODISCARD PtrInterval ptr_interval() const {
            return build_interval(ptr_t(ptr<const std::byte *>()),
                                  ptr_t(ptr<const std::byte *>() + size_in_bytes()));
        }

        void download(std::remove_const_t<T> *host_ptr, size_t n_elements = -1, size_t offset = 0) {
            DCHECK(valid());
            n_elements = fix_count(offset, n_elements, size());
            _impl->download(host_ptr, n_elements * sizeof(T), offset * sizeof(T));
        }

        void download_async(Dispatcher &dispatcher, std::remove_const_t<T> *host_ptr,
                            size_t n_elements = -1, size_t offset = 0) {
            DCHECK(valid());
            n_elements = fix_count(offset, n_elements, size());
            _impl->download_async(dispatcher, host_ptr, n_elements * sizeof(T), offset * sizeof(T));
        }

        void upload(const T *host_ptr, size_t n_elements = -1, size_t offset = 0) {
            DCHECK(valid());
            n_elements = fix_count(offset, n_elements, size());
            _impl->upload(host_ptr, n_elements * sizeof(T), offset * sizeof(T));
        }

        void upload_async(Dispatcher &dispatcher, const T *host_ptr, size_t n_elements = -1, size_t offset = 0) {
            DCHECK(valid());
            n_elements = fix_count(offset, n_elements, size());
            _impl->upload_async(dispatcher, host_ptr, n_elements * sizeof(T), offset * sizeof(T));
        }
    }; // luminous::Buffer
} // luminous