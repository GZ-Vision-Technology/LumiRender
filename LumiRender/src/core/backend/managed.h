//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "buffer.h"
#include "device.h"
#include "core/concepts.h"

namespace luminous {
    template<typename T, typename U = const std::remove_const_t<T>>
    struct Managed : public Noncopyable, public std::vector<T> {
    public:
        using BaseClass = std::vector<T>;
        using THost = T;
        using TDevice = U;
    private:
        static_assert(!std::is_pointer_v<std::remove_pointer_t<THost>>, "THost can not be the secondary pointer!");
        Buffer <TDevice> _device_buffer{nullptr};
    public:
        Managed() = default;

        size_t size_in_bytes() const {
            return BaseClass::size() * sizeof(T);
        }

        void reset(const vector <THost> &v) {
            BaseClass::reserve(v.capacity());
            BaseClass::resize(v.size());
            std::memcpy(BaseClass::data(), v.data(), sizeof(THost) * v.size());
        }

        void reset(const vector <THost> &v, const SP <Device> &device) {
            _device_buffer = device->allocate_buffer<TDevice>(v.size());
            reset(v);
        }

        void reset(THost *host, int n = 1) {
            BaseClass::reserve(n);
            BaseClass::resize(n);
            std::memcpy(BaseClass::data(), host, sizeof(THost) * n);
        }

        void reset(THost *host, const SP <Device> &device, int n = 1) {
            _device_buffer = device->allocate_buffer<TDevice>(n);
            reset(host, n);
        }

        void reset(const SP <Device> &device, size_t n) {
            BaseClass::resize(n);
            _device_buffer = device->allocate_buffer<TDevice>(n);
            std::memset(BaseClass::data(), 0, sizeof(THost) * n);
        }

        void append(const std::vector<THost> &v) {
            BaseClass::insert(BaseClass::cend(), v.cbegin(), v.cend());
        }

        void allocate_device(const SP <Device> device, size_t size = 0) {
            size = size == 0 ? BaseClass::size() : size;
            if (size == 0) {
                return;
            }
            _device_buffer = device->allocate_buffer<TDevice>(size);
        }

        NDSC BufferView <const THost> const_host_buffer_view(size_t offset = 0, size_t count = -1) const {
            count = fix_count(offset, count, BaseClass::size());
            return BufferView<const THost>(((const THost *) BaseClass::data()) + offset, count);
        }

        NDSC BufferView <THost> host_buffer_view(size_t offset = 0, size_t count = -1) const {
            count = fix_count(offset, count, BaseClass::size());
            return BufferView<THost>(((THost *) BaseClass::data()) + offset, count);
        }

        NDSC BufferView <TDevice> device_buffer_view(size_t offset = 0, size_t count = -1) const {
            return _device_buffer.view(offset, count);
        }

        NDSC BufferView <const Distribution1D> const_device_buffer_view(size_t offset = 0, size_t count = -1) const {
            return _device_buffer.view(offset, count);
        }

        const Buffer <TDevice> &device_buffer() const {
            return _device_buffer;
        }

        void clear() {
            clear_device();
            clear_host();
        }

        void clear_host() {
            BaseClass::clear();
        }

        void clear_device() {
            _device_buffer.clear();
        }

        template<typename pointer_type = void *>
        pointer_type device_ptr() const {
            return _device_buffer.template ptr<pointer_type>();
        }

        TDevice *device_data() const {
            return _device_buffer.data();
        }

        const THost *operator->() const {
            return BaseClass::data();
        }

        THost *operator->() {
            return BaseClass::data();
        }

        void synchronize_to_gpu() {
            if (BaseClass::size() == 0) {
                return;
            }
            _device_buffer.upload(BaseClass::data());
        }

        void synchronize_to_cpu() {
            if (BaseClass::size() == 0) {
                return;
            }
            _device_buffer.download(BaseClass::data());
        }
    };
}