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
#include "graphics/header.h"

namespace luminous {

    class Dispatcher;

    class RawBuffer {
    public:
        class Impl {
        public:
            virtual void download(void *host_ptr, size_t size = 0, size_t offset = 0) = 0;

            virtual void download_async(Dispatcher &dispatcher, void *host_ptr, size_t size = 0, size_t offset = 0) = 0;

            virtual void upload(const void *host_ptr, size_t size = 0, size_t offset = 0) = 0;

            virtual void
            upload_async(Dispatcher &dispatcher, const void *host_ptr, size_t size = 0, size_t offset = 0) = 0;

            virtual void *address(size_t offset = 0) const = 0;

            virtual size_t size() const = 0;

            virtual void *ptr() = 0;

            virtual ~Impl() = default;
        };

        RawBuffer(std::unique_ptr<Impl> impl) : _impl(std::move(impl)) {}

        Impl *impl_mut() const {
            assert(valid());
            return _impl.get();
        }

        bool valid() const {
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

    template<typename T>
    struct BufferView {
    private:
        // point to device memory or host memory
        T *_ptr;
        // count of elements
        const size_t _num;
    public:
        using value_type = T;
        using iterator = T *;
        using const_iterator = const T *;

        XPU BufferView() : _ptr(nullptr), _num(0) {}

        XPU BufferView(T *ptr, size_t num)
                : _ptr(ptr), _num(num) {}

        XPU BufferView(std::initializer_list<value_type> v)
                : BufferView(v.begin(), v.size()) {}

        NDSC_XPU iterator begin() { return _ptr; }

        NDSC_XPU iterator end() { return _ptr + _num; }

        NDSC_XPU const_iterator c_begin() const { return _ptr; }

        NDSC_XPU const_iterator c_end() const { return _ptr + _num; }

        NDSC_XPU T operator[](size_t i) {
            DCHECK_LT(i, size());
            return _ptr[i];
        }

        NDSC_XPU bool empty() const { return _num == 0; }

        NDSC_XPU T front() const { return _ptr[0]; }

        NDSC_XPU T back() const { return _ptr[_num - 1]; }

        NDSC_XPU BufferView sub_view(size_t pos, size_t count) {
            size_t np = count < (size() - pos) ? count : (size() - pos);
            return BufferView(_ptr + pos, np);
        }

        NDSC_XPU size_t size() const {
            return _num;
        }
    };

    template<class T = std::byte, typename TP = void *>
    class Buffer : public RawBuffer {
    public:
        using value_type = T;
        using pointer_type = TP;

        using RawBuffer::RawBuffer;

        Buffer(RawBuffer buf) : RawBuffer(std::move(buf)) {}

        value_type *data() const { return reinterpret_cast<value_type *>(ptr()); }

        size_t stride_in_bytes() const { return sizeof(value_type); }

        template<typename U = pointer_type>
        auto ptr() const {
            DCHECK(valid());
            return (U) _impl->ptr();
        }

        BufferView<T> view() const {
            return BufferView<T>(ptr<T>(), size());
        }

        template<typename U = void *>
        auto address(size_t offset = 0) const {
            DCHECK(valid());
            return (U) _impl->address(offset * sizeof(value_type));
        }

        size_t size() const {
            DCHECK(valid());
            return _impl->size() / sizeof(value_type);
        }

        size_t size_in_bytes() const {
            DCHECK(valid());
            return _impl->size();
        }

        void download(T *host_ptr, size_t n_elements = 0, size_t offset = 0) {
            DCHECK(valid());
            n_elements = n_elements == 0 ? _impl->size() / sizeof(T) : n_elements;
            DCHECK(offset * sizeof(T) + n_elements * sizeof(T) <= _impl->size());
            _impl->download(host_ptr, n_elements * sizeof(T), offset * sizeof(T));
        }

        void download_async(Dispatcher &dispatcher, T *host_ptr, size_t n_elements = 0, size_t offset = 0) {
            DCHECK(valid());
            n_elements = n_elements == 0 ? _impl->size() / sizeof(T) : n_elements;
            DCHECK(offset * sizeof(T) + n_elements * sizeof(T) <= _impl->size());
            _impl->download_async(dispatcher, host_ptr, n_elements * sizeof(T), offset * sizeof(T));
        }

        void upload(const T *host_ptr, size_t n_elements = 0, size_t offset = 0) {
            DCHECK(valid());
            n_elements = n_elements == 0 ? _impl->size() / sizeof(T) : n_elements;
            DCHECK(offset * sizeof(T) + n_elements * sizeof(T) <= _impl->size());
            _impl->upload(host_ptr, n_elements * sizeof(T), offset * sizeof(T));
        }

        void upload_async(Dispatcher &dispatcher, const T *host_ptr, size_t n_elements = 0, size_t offset = 0) {
            DCHECK(valid());
            n_elements = n_elements == 0 ? _impl->size() / sizeof(T) : n_elements;
            DCHECK(offset * sizeof(T) + n_elements * sizeof(T) <= _impl->size());
            _impl->upload_async(dispatcher, host_ptr, n_elements * sizeof(T), offset * sizeof(T));
        }
    }; // luminous::Buffer
} // luminous