//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "buffer.h"
#include "device.h"
#include "core/concepts.h"

namespace luminous {
    template<typename T, typename U = const std::remove_const_t<T>>
    struct Managed : Noncopyable {
    public:
        using THost = T;
        using TDevice = U;
    private:
        static_assert(!std::is_pointer_v<std::remove_pointer_t<THost>>, "THost can not be the secondary pointer!");
        std::vector<THost> _host{};
        Buffer <TDevice> _device_buffer{nullptr};
    public:
        Managed() = default;

        Managed(THost *host, const SP <Device> &device, int n = 1)
                : _host(host) {
            _device_buffer = device->allocate_buffer<TDevice>(n);
        }

        Managed(Managed<THost, TDevice> &&other) {
            _host = move(other._host);
            _device_buffer = move(other._device_buffer);
        }

        size_t size_in_bytes() const {
            return _device_buffer.size_in_bytes();
        }

        size_t size() const {
            return _host.size();
        }

        void reset(THost *host, const SP <Device> &device, int n = 1) {
            _device_buffer = device->allocate_buffer<TDevice>(n);
            _host.reserve(n);
            _host.resize(n);
            std::memcpy(_host.data(), host, sizeof(THost) * n);
        }

        void reset(THost *host, int n = 1) {
            _host.reserve(n);
            _host.resize(n);
            std::memcpy(_host.data(), host, sizeof(THost) * n);
        }

        void reset(std::vector<THost> v) {
            _n_elements = v.size();
            _host = std::move(v);
        }

        void append(const std::vector<THost> &v) {
            _host.insert(_host.cend(), v.cbegin(), v.cend());
        }

        void reset(std::vector<THost> v, const SP <Device> &device) {
            _host = std::move(v);
            _device_buffer = device->allocate_buffer<TDevice>(_host.size());
        }

        void reset(size_t n) {
            _host.resize(n);
            std::memset(_host.data(), 0, sizeof(THost) * n);
        }

        void reset(const SP <Device> &device, size_t n) {
            _host.resize(n);
            _device_buffer = device->allocate_buffer<TDevice>(n);
            std::memset(_host.data(), 0, sizeof(THost) * n);
        }

        template<typename... Args>
        void reset_all(Args &&...args) {
            reset(std::forward<Args>(args)...);
            synchronize_to_gpu();
        }

        void allocate_device(const SP <Device> device) {
            _device_buffer = device->allocate_buffer<TDevice>(_host.size());
        }

        BufferView <TDevice> host_buffer_view(size_t offset = 0, size_t count = 0) const {
            count = fix_count(offset, count, size());
            return BufferView<TDevice>(((TDevice *) _host.data()) + offset, count);
        }

        BufferView <TDevice> device_buffer_view(size_t offset = 0, size_t count = 0) const {
            return _device_buffer.view(offset, count);
        }

        const Buffer <TDevice> &device_buffer() const {
            return _device_buffer;
        }

        void clear() {
            _device_buffer.clear();
            _host.clear();
        }

        void clear_host() {
            _host.clear();
        }

        void clear_device() {
            _device_buffer.clear();
        }

        const std::vector<THost> &c_vector() const {
            return _host;
        }

        std::vector<THost> &vector() {
            return _host;
        }

        THost *get() {
            return reinterpret_cast<THost *>(_host.data());
        }

        template<typename pointer_type = void *>
        pointer_type device_ptr() const {
            return _device_buffer.template ptr<pointer_type>();
        }

        TDevice *device_data() const {
            return _device_buffer.data();
        }

        template<typename Index>
        THost &operator[](Index i) {
            DCHECK(i < _device_buffer.size());
            return _host[i];
        }

        template<typename Index>
        const THost &operator[](Index i) const {
            DCHECK(i < _device_buffer.size());
            return _host[i];
        }

        THost *operator->() {
            return _host.data();
        }

        void synchronize_to_gpu() {
            _device_buffer.upload(_host.data());
        }

        void synchronize_to_cpu() {
            _device_buffer.download(_host.data());
        }
    };
}