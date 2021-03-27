//
// Created by Zero on 2021/3/24.
//


#pragma once

#include "buffer.h"
#include "device.h"
#include "core/concepts.h"

namespace luminous {
    template<typename THost, typename TDevice = std::remove_pointer_t<THost>>
    struct Managed_old : Noncopyable {
    private:
        static_assert(!std::is_pointer_v<std::remove_pointer_t<THost>>, "THost can not be the secondary pointer!");
        THost _host{};
        Buffer <TDevice> _device_buffer{nullptr};
    public:
        Managed_old(THost host, const SP <Device> &device, int n = 1)
                : _host(host) {
            _device_buffer = device->allocate_buffer<TDevice>(n);
        }

        Managed_old() {}

        void reset(THost host, const SP <Device> &device, int n = 1) {
            _device_buffer = device->allocate_buffer<TDevice>(n);
            _host = host;
        }

        const Buffer <TDevice> &device_buffer() const {
            return _device_buffer;
        }

        THost get() {
            return _host;
        }

        template<typename T = void *>
        T device_ptr() const {
            return _device_buffer.ptr<T>();
        }

        auto device_data() const {
            return _device_buffer.data();
        }

        auto operator[](uint i) {
            static_assert(std::is_pointer_v<THost>, "subscript only operate pointer!");
            assert(i < _device_buffer.size());
            return _host[i];
        }

        auto operator->() {
            if constexpr (std::is_pointer_v<THost>) {
                return _host;
            } else {
                return &_host;
            }
        }

        void synchronize_to_gpu() {
            if constexpr (std::is_pointer_v<THost>) {
                _device_buffer.upload(_host);
            } else {
                _device_buffer.upload(&_host);
            }
        }

        void synchronize_to_cpu() {
            if constexpr (std::is_pointer_v<THost>) {
                _device_buffer.download(_host);
            } else {
                _device_buffer.download(&_host);
            }
        }
    };

    template<typename THost, typename TDevice = THost>
    struct Managed : Noncopyable {
    private:
        static_assert(!std::is_pointer_v<std::remove_pointer_t<THost>>, "THost can not be the secondary pointer!");
        size_t _n_elements{0};
        UP<THost[]> _host{};
        Buffer <TDevice> _device_buffer{nullptr};
    public:
        Managed(THost *host, const SP <Device> &device, int n = 1)
                : _host(host) {
            _device_buffer = device->allocate_buffer<TDevice>(n);
        }

        Managed() {}

        void reset(THost *host, const SP <Device> &device, int n = 1) {
            _n_elements = n;
            _device_buffer = device->allocate_buffer<TDevice>(_n_elements);
            _host.reset(new THost[n]);
            std::memcpy(_host.get(), host, sizeof(THost) * _n_elements);
        }

        void reset(const std::vector<THost> &v, const SP<Device> &device) {
            _n_elements = v.size();
            _device_buffer = device->allocate_buffer<TDevice>(_n_elements);
            _host.reset(new THost[_n_elements]);
            std::memcpy(_host.get(), v.data(), sizeof(THost) * _n_elements);
        }

        const Buffer <TDevice> &device_buffer() const {
            return _device_buffer;
        }

        void clear() {
            _device_buffer.clear();
            _host.reset(nullptr);
        }

        THost get() {
            return _host.get();
        }

        template<typename T = void *>
        T device_ptr() const {
            return _device_buffer.ptr<T>();
        }

        TDevice device_data() const {
            return _device_buffer.data();
        }

        auto operator[](uint i) {
            assert(i < _device_buffer.size());
            return _host.get()[i];
        }

        auto operator->() {
            return _host.get();
        }

        void synchronize_to_gpu() {
            _device_buffer.upload(_host.get());
        }

        void synchronize_to_cpu() {
            _device_buffer.download(_host.get());
        }
    };
}