//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "buffer.h"
#include "device.h"
#include "core/concepts.h"

namespace luminous {
    template<typename THost, typename TDevice = THost>
    struct Managed : Noncopyable {
    private:
        static_assert(!std::is_pointer_v<std::remove_pointer_t<THost>>, "THost can not be the secondary pointer!");
        size_t _n_elements{0};
        std::vector<THost> _host{};
        Buffer <TDevice> _device_buffer{nullptr};
    public:
        Managed(THost *host, const SP <Device> &device, int n = 1)
                : _host(host) {
            _device_buffer = device->allocate_buffer<TDevice>(n);
        }

        Managed() {}

        size_t size_in_bytes() const {
            return _device_buffer.size_in_bytes();
        }

        size_t size() const {
            return _n_elements;
        }

        void reset(THost *host, const SP <Device> &device, int n = 1) {
            _n_elements = n;
            _device_buffer = device->allocate_buffer<TDevice>(_n_elements);
            _host.reserve(_n_elements);
            _host.resize(_n_elements);
            std::memcpy(_host.data(), host, sizeof(THost) * _n_elements);
        }

        void reset(THost *host, int n = 1) {
            _n_elements = n;
            _host.reserve(_n_elements);
            _host.resize(_n_elements);
            std::memcpy(_host.data(), host, sizeof(THost) * _n_elements);
        }

        void reset(std::vector<THost> v) {
            _n_elements = v.size();
            _host = std::move(v);
        }

        void reset(std::vector<THost> v, const SP <Device> &device) {
            _n_elements = v.size();
            _host = std::move(v);
            _device_buffer = device->allocate_buffer<TDevice>(_n_elements);
        }

        void reset(size_t n) {
            _n_elements = n;
            _host.resize(n);
            std::memset(_host.data(), 0, sizeof(THost) * _n_elements);
        }

        void reset(const SP <Device> &device, size_t n) {
            _n_elements = n;
            _host.resize(n);
            _device_buffer = device->allocate_buffer<TDevice>(_n_elements);
            std::memset(_host.data(), 0, sizeof(THost) * _n_elements);
        }

        template<typename... Args>
        void reset_all(Args &&...args) {
            reset(std::forward<Args>(args)...);
            synchronize_to_gpu();
        }

        BufferView<THost> host_buffer_view() const {
            return BufferView<THost>(_host.data(), _host.size());
        }

        BufferView<TDevice> device_buffer_view() const {
            return _device_buffer.view();
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
            return _host.data();
        }

        template<typename T = void *>
        T device_ptr() const {
            return _device_buffer.ptr<T>();
        }

        TDevice *device_data() const {
            return _device_buffer.data();
        }

        template<typename Index>
        THost &operator[](Index i) {
            assert(i < _device_buffer.size());
            return _host[i];
        }

        template<typename Index>
        const THost &operator[](Index i) const {
            assert(i < _device_buffer.size());
            return _host[i];
        }

        THost* operator->() {
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